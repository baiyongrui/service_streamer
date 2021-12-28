import os
import threading
import logging
import time
import uuid
import weakref
from queue import Queue, Empty
from typing import List
import asyncio

import torch


TIMEOUT = 1
TIME_SLEEP = 0.001
WORKER_TIMEOUT = 20
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


class ThreadSafeEvent(asyncio.Event):
    def set(self):
        #FIXME: The _loop attribute is not documented as public api!
        self._loop.call_soon_threadsafe(super().set)


class Future(object):
    def __init__(self, task_id, task_size, future_cache_ref, loop=None):
        self._id = task_id
        self._size = task_size
        self._future_cache_ref = future_cache_ref
        self._outputs = []
        # self._finish_event = threading.Event()
        self._finish_event = ThreadSafeEvent(loop=loop)


    async def result(self):
        if self._size == 0:
            self._finish_event.set()
            return []
        # finished = self._finish_event.wait(timeout)
        await self._finish_event.wait()

        # if not finished:
        #     raise TimeoutError("Task: %d Timeout" % self._id)

        # remove from future_cache
        future_cache = self._future_cache_ref()
        if future_cache is not None:
            del future_cache[self._id]

        # [(request_id, output), ...] sorted by request_id
        self._outputs.sort(key=lambda i: i[0])
        # restore batch result from outputs
        batch_result = [i[1] for i in self._outputs]

        return batch_result

    def done(self):
        if self._finish_event.is_set():
            return True

    def _append_result(self, it_id, it_output):
        self._outputs.append((it_id, it_output))
        if len(self._outputs) >= self._size:
            self._finish_event.set()


class _FutureCache(dict):
    "Dict for weakref only"
    pass


class _BaseStreamer(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._client_id = str(uuid.uuid4())
        self._task_id = 0
        self._future_cache = _FutureCache()  # {task_id: future}
        self._worker_timeout = kwargs.get("worker_timeout", WORKER_TIMEOUT)

        self.back_thread = threading.Thread(target=self._loop_collect_result, name="thread_collect_result")
        self.back_thread.daemon = True
        self.lock = threading.Lock()

    def _delay_setup(self):
        self.back_thread.start()

    def _send_request(self, task_id, request_id, model_input):
        raise NotImplementedError

    def _recv_response(self, timeout=TIMEOUT):
        raise NotImplementedError

    def _input(self, batch: List) -> int:
        """
        input a batch, distribute each item to mq, return task_id
        """
        # task id in one client
        self.lock.acquire()
        task_id = self._task_id
        self._task_id += 1
        self.lock.release()
        # request id in one task
        request_id = 0

        future = Future(task_id, len(batch), weakref.ref(self._future_cache))
        self._future_cache[task_id] = future

        for model_input in batch:
            self._send_request(task_id, request_id, model_input)
            request_id += 1

        return task_id

    def _loop_collect_result(self):
        # logger.info("start _loop_collect_result")
        while True:
            message = self._recv_response(timeout=TIMEOUT)
            if message:
                (task_id, request_id, item) = message
                future = self._future_cache[task_id]
                future._append_result(request_id, item)
            else:
                # todo
                time.sleep(TIME_SLEEP)

    def _output(self, task_id: int) -> List:
        future = self._future_cache[task_id]
        batch_result = future.result(self._worker_timeout)
        return batch_result

    def submit(self, batch):
        task_id = self._input(batch)
        future = self._future_cache[task_id]
        return future

    def predict(self, batch):
        task_id = self._input(batch)
        ret = self._output(task_id)
        assert len(batch) == len(ret), "input batch size {} and output batch size {} must be equal.".format(len(batch), len(ret))
        return ret

    def destroy_workers(self):
        raise NotImplementedError


class _BaseStreamWorker(object):
    def __init__(self, predict_function, batch_size, max_latency, *args, **kwargs):
        super().__init__()
        assert callable(predict_function)
        self._pid = os.getpid()
        self._predict = predict_function
        self._batch_size = batch_size
        self._max_latency = max_latency
        self._destroy_event = kwargs.get("destroy_event", None)

    def run_forever(self, *args, **kwargs):
        self._pid = os.getpid()  # overwrite the pid
        logger.info("[gpu worker %d] %s start working" % (self._pid, self))

        while True:
            handled = self._run_once()
            if self._destroy_event and self._destroy_event.is_set():
                break
            if not handled:
                # sleep if no data handled last time
                time.sleep(TIME_SLEEP)
        logger.info("[gpu worker %d] %s shutdown" % (self._pid, self))

    def model_predict(self, batch_input):
        batch_result = self._predict(batch_input)
        assert len(batch_input) == len(batch_result), "input batch size {} and output batch size {} must be equal.".format(len(batch_input), len(batch_result))
        return batch_result

    def _run_once(self):
        batch = []
        start_time = time.time()
        for i in range(self._batch_size):
            try:
                item = self._recv_request(timeout=self._max_latency)
            except TimeoutError:
                # each item timeout exceed the max latency
                break
            else:
                batch.append(item)
            if (time.time() - start_time) > self._max_latency:
                # total batch time exceeds the max latency
                break
        if not batch:
            return 0

        model_inputs = [i[3] for i in batch]
        model_outputs = self.model_predict(model_inputs)

        # publish results to redis
        for i, item in enumerate(batch):
            client_id, task_id, request_id, _ = item
            self._send_response(client_id, task_id, request_id, model_outputs[i])

        batch_size = len(batch)
        logger.info("[gpu worker %d] run_once batch_size: %d start_at: %s spend: %s" % (
            self._pid, batch_size, start_time, time.time() - start_time), flush=True)
        return batch_size

    def _recv_request(self, timeout=TIMEOUT):
        raise NotImplementedError

    def _send_response(self, client_id, task_id, request_id, model_input):
        raise NotImplementedError


class PredictWorker(object):
    def __init__(self, predict_function, batch_size, max_latency, request_queue, response_queue, *args, **kwargs):
        super().__init__()
        assert callable(predict_function)
        self._pid = os.getpid()
        self._predict = predict_function
        self._batch_size = batch_size
        self._max_latency = max_latency
        self._destroy_event = kwargs.get("destroy_event", None)

        self._request_queue = request_queue
        self._response_queue = response_queue

    def run_forever(self, *args, **kwargs):
        self._pid = os.getpid()  # overwrite the pid
        logger.info("[gpu worker %d] %s start working" % (self._pid, self))

        while True:
            handled = self._run_once()
            if self._destroy_event and self._destroy_event.is_set():
                break
            if not handled:
                # sleep if no data handled last time
                time.sleep(TIME_SLEEP)
        logger.info("[gpu worker %d] %s shutdown" % (self._pid, self))

    def model_predict(self, batch_input):
        err = None
        try:
            batch_result = self._predict(batch_input)
        except RuntimeError as e:
            err = e
            batch_result = [None for _ in range(len(batch_input))]
            logger.error("[gpu worker %d] model_predict error: %s" % (self._pid, e))

            # FIXME clean GPU cache?
            if 'CUDA out of memory' in str(e):
                torch.cuda.empty_cache()

        if len(batch_input) != len(batch_result):
            err = RuntimeError("input batch size {} and output batch size {} must be equal.".format(len(batch_input), len(batch_result)))
            batch_result = [None for _ in range(len(batch_input))]

        return batch_result, err

    def _run_once(self):
        batch = []
        start_time = time.time()
        for i in range(self._batch_size):
            try:
                item = self._recv_request(timeout=self._max_latency)
            except TimeoutError:
                # each item timeout exceed the max latency
                break
            else:
                batch.append(item)
            if (time.time() - start_time) > self._max_latency:
                # total batch time exceeds the max latency
                break
        if not batch:
            return 0

        model_inputs = [i[3] for i in batch]
        model_outputs, err = self.model_predict(model_inputs)

        # Prepare for dispatch results
        for i, item in enumerate(batch):
            client_id, task_id, request_id, _ = item
            predict_result = PredictResult(model_outputs[i], err)
            self._send_response(client_id, task_id, request_id, predict_result)

        batch_size = len(batch)
        logger.info("[gpu worker %d] run_once batch_size: %d start_at: %s spend: %s" % (
            self._pid, batch_size, start_time, time.time() - start_time))
        return batch_size

    def _recv_request(self, timeout=TIMEOUT):
        try:
            item = self._request_queue.get(timeout=timeout)
        except Empty:
            raise TimeoutError
        else:
            return item

    def _send_response(self, client_id, task_id, request_id, model_output):
        self._response_queue.put((task_id, request_id, model_output))


class ThreadedStreamer(_BaseStreamer):
    def __init__(self, predict_function, batch_size, max_latency=0.1, worker_timeout=WORKER_TIMEOUT):
        super().__init__(worker_timeout=worker_timeout)
        self._input_queue = Queue()
        self._output_queue = Queue()
        self._worker_destroy_event=threading.Event()
        # self._worker = ThreadedWorker(predict_function, batch_size, max_latency,
        #                               self._input_queue, self._output_queue,
        #                               destroy_event=self._worker_destroy_event)
        self._worker = PredictWorker(predict_function, batch_size, max_latency,
                                      self._input_queue, self._output_queue,
                                      destroy_event=self._worker_destroy_event)
        self._worker_thread = threading.Thread(target=self._worker.run_forever, name="thread_worker")
        self._worker_thread.daemon = True
        self._worker_thread.start()
        self._delay_setup()

    def _send_request(self, task_id, request_id, model_input):
        self._input_queue.put((0, task_id, request_id, model_input))

    def _recv_response(self, timeout=TIMEOUT):
        try:
            message = self._output_queue.get(timeout=timeout)
        except Empty:
            message = None
        return message

    def destroy_workers(self):
        self._worker_destroy_event.set()
        self._worker_thread.join(timeout=self._worker_timeout)
        if self._worker_thread.is_alive():
            raise TimeoutError("worker_thread destroy timeout")
        logger.info("workers destroyed")


class ThreadedWorker(_BaseStreamWorker):
    def __init__(self, predict_function, batch_size, max_latency, request_queue, response_queue, *args, **kwargs):
        super().__init__(predict_function, batch_size, max_latency, *args, **kwargs)
        self._request_queue = request_queue
        self._response_queue = response_queue

    def _recv_request(self, timeout=TIMEOUT):
        try:
            item = self._request_queue.get(timeout=timeout)
        except Empty:
            raise TimeoutError
        else:
            return item

    def _send_response(self, client_id, task_id, request_id, model_output):
        self._response_queue.put((task_id, request_id, model_output))


class Streamer(_BaseStreamer):
    def __init__(self, predict_function_or_model, batch_size, max_latency=0.1, worker_num=1,
                 cuda_devices=None, model_init_args=None, model_init_kwargs=None, wait_for_worker_ready=False,
                 mp_start_method='spawn', worker_timeout=WORKER_TIMEOUT):
        super().__init__(worker_timeout=worker_timeout)
        self.worker_num = worker_num
        self.cuda_devices = cuda_devices
        self.mp = multiprocessing.get_context(mp_start_method)
        self._input_queue = self.mp.Queue()
        self._output_queue = self.mp.Queue()
        self._worker = StreamWorker(predict_function_or_model, batch_size, max_latency,
                                    self._input_queue, self._output_queue,
                                    model_init_args, model_init_kwargs)
        self._worker_ps = []
        self._worker_ready_events = []
        self._worker_destroy_events = []
        self._setup_gpu_worker()
        if wait_for_worker_ready:
            self._wait_for_worker_ready()
        self._delay_setup()

    def _setup_gpu_worker(self):
        for i in range(self.worker_num):
            ready_event = self.mp.Event()
            destroy_event = self.mp.Event()
            if self.cuda_devices is not None:
                gpu_id = self.cuda_devices[i % len(self.cuda_devices)]
                args = (gpu_id, ready_event, destroy_event)
            else:
                args = (None, ready_event, destroy_event)
            p = self.mp.Process(target=self._worker.run_forever, args=args, name="stream_worker", daemon=True)
            p.start()
            self._worker_ps.append(p)
            self._worker_ready_events.append(ready_event)
            self._worker_destroy_events.append(destroy_event)

    def _wait_for_worker_ready(self, timeout=None):
        if timeout is None:
            timeout = self._worker_timeout
        # wait for all workers finishing init
        for (i, e) in enumerate(self._worker_ready_events):
            # todo: select all events with timeout
            is_ready = e.wait(timeout)
            logger.info("gpu worker:%d ready state: %s" % (i, is_ready))

    def _send_request(self, task_id, request_id, model_input):
        self._input_queue.put((0, task_id, request_id, model_input))

    def _recv_response(self, timeout=TIMEOUT):
        try:
            message = self._output_queue.get(timeout=timeout)
        except Empty:
            message = None
        return message

    def destroy_workers(self):
        for e in self._worker_destroy_events:
            e.set()
        for p in self._worker_ps:
            p.join(timeout=self._worker_timeout)
            if p.is_alive():
                raise TimeoutError("worker_process destroy timeout")
        logger.info("workers destroyed")


class StreamWorker(_BaseStreamWorker):
    def __init__(self, predict_function_or_model, batch_size, max_latency, request_queue, response_queue,
                 model_init_args, model_init_kwargs, *args, **kwargs):
        super().__init__(predict_function_or_model, batch_size, max_latency, *args, **kwargs)
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._model_init_args = model_init_args or []
        self._model_init_kwargs = model_init_kwargs or {}

    def run_forever(self, gpu_id=None, ready_event=None, destroy_event=None):
        # if it is a managed model, lazy init model after forked & set CUDA_VISIBLE_DEVICES
        if isinstance(self._predict, type) and issubclass(self._predict, ManagedModel):
            model_class = self._predict
            logger.info("[gpu worker %d] init model on gpu:%s" % (os.getpid(), gpu_id))
            self._model = model_class(gpu_id)
            self._model.init_model(*self._model_init_args, **self._model_init_kwargs)
            logger.info("[gpu worker %d] init model on gpu:%s" % (os.getpid(), gpu_id))
            self._predict = self._model.predict
            if ready_event:
                ready_event.set()  # tell father process that init is finished
        if destroy_event:
            self._destroy_event = destroy_event
        super().run_forever()

    def _recv_request(self, timeout=TIMEOUT):
        try:
            item = self._request_queue.get(timeout=timeout)
        except Empty:
            raise TimeoutError
        else:
            return item

    def _send_response(self, client_id, task_id, request_id, model_output):
        self._response_queue.put((task_id, request_id, model_output))


class PredictResult(object):
    def __init__(self, result, error=None):
        self.result = result
        self.error = error

    def err_desc(self):
        if self.error is None:
            return ""
        return str(self.error)