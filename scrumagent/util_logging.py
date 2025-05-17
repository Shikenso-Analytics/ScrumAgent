"""
This module is part of the kensoutil set.
Current version is: 0.2.0
"""
import json
import logging
import logging.handlers
import logging.config
import asyncio
import os
import queue

from functools import wraps
from queue import SimpleQueue
from typing import List, Callable, Tuple, Type, Optional, Union
from httpx import HTTPError

# The reason why a SimpleQueue is used instead of Queue is reentrancy.
# For more detail look at the std documentation: https://docs.python.org/3.8/library/queue.html#simplequeue-objects
_safe = SimpleQueue()

# This List is later accessed and manipulated from the outside
_handlers: List[logging.Handler] = []


class DiscordChannelLogger(logging.Handler):
    """
    This class provides an abstraction for logging to a Discord channel.
    """

    def __init__(self, record_queue: queue.Queue) -> None:
        logging.Handler.__init__(self)
        self.subject = "Discord SCRUM Agent Error"
        self._record_queue = record_queue
        self.discord_channel_ids = []

    def override_discord_channel_id(self, discord_channel_ids: List) -> List:
        self.discord_channel_ids = discord_channel_ids
        return self.discord_channel_ids

    def add_recipient(self, additional_discord_channels: List) -> List:
        self.discord_channel_ids.extend(additional_discord_channels)
        return self.discord_channel_ids

    def change_subject(self, new_subject: str) -> str:
        """This function is a setter for the value 'self.subject'.

        :param new_subject: Set the new subject.
        :type new_subject: str
        :return: Returns the new set subject.
        :rtype: str
        """
        self.subject = new_subject
        return new_subject

    def emit(self, record: logging.LogRecord) -> None:
        """This function is called from the Logger and is the actual log process.

        :param record: The event to log.
        :type record: logging.LogRecord
        """

        rec = self.format(record=record)
        self._record_queue.put_nowait((self.subject, rec, self.discord_channel_ids))


discord_log_queue = queue.Queue()
discord_logger_handler = DiscordChannelLogger(record_queue=discord_log_queue)

# set logging levels
discord_logger_handler.setLevel(30)
# fly.setLevel(10)

# add default handlers to handler List
# _handlers.append(fly)
_handlers.append(discord_logger_handler)


def override_defaults(
    override: List = None,
    addreciever: List = None,
    subject: str = None,
    log_file_path: str = None,
    log_file_name: str = None,
) -> bool:
    """Only use this function to override parameters if you dont not alter the position
    of the std. SMTPHandler and std. FileHandler inside the 'handlers' List.

    :param override:A List of strings representing the discord channels to contact
    :type override: List[str], optional
    :param addreciever: Add discord channels to the default channels, defaults to None
    :type addreciever: List[str], optional
    :param subject: Change the default subject, defaults to None
    :type subject: str, optional
    :param log_file_path:   Alter the std. log-file position, for changes provide and
                            abs. path. Std. is os.getcwd(), the current working directory,
                            defaults to None
    :type log_file_path: str, optional
    :param log_file_name: Provide an optional log-file name, std. is Errors.log, defaults to None
    :type log_file_name: str, optional
    :return: Returns True if all parameters could be processed.
    :rtype: bool
    """
    if override is not None:
        _handlers[0].override_discord_channel_id(override)
    if addreciever is not None:
        _handlers[0].add_recipient(addreciever)
    if subject is not None:
        _handlers[0].change_subject(subject)
    return True


class Logwriter(logging.handlers.QueueHandler):
    """This is a small abstraction class to handel asyncio errors."""

    def emit(self, record: logging.LogRecord) -> None:
        """Mutes errors in the logging system itself.

        :param record: The event to be logged.
        :type record: logging.LogRecord
        """
        try:
            self.enqueue(record=record)
        except asyncio.CancelledError:
            self.handleError(record=record)


def start_listener(
    handlers: List[logging.Handler] = _handlers,
) -> logging.handlers.QueueListener:
    """This Listener has only to be inited once per python-interpreter.
    Invoke this command after all Handlers are setup to prevent possible problems.
    If some Handlers are not processed -> check shallow copy of function parameter.


    :param handlers: Logging handlers, defaults to _handlers
    :type handlers: List[logging.Handler], optional
    :return: This Handler watches the given Queue for new messages.
    :rtype: logging.handlers.QueueListener
    """
    listener = logging.handlers.QueueListener(
        _safe, *handlers, respect_handler_level=True
    )
    listener.start()
    return listener


# create Module Logger
def init_module_logger(
    name: str = __name__,
) -> logging.Logger:
    """This function creates an Logger for the given Module.
    And adds QueueHandler with the SimpleQueue.

    :param name: Name of the logger, defaults to __name__
    :type name: str, optional
    :return: Fully setup Logger for futher processing.
    :rtype: logging.Logger
    """
    logger = logging.getLogger(name)
    logger.addHandler(Logwriter(_safe))
    return logger


def another_handler(handler: logging.Handler) -> bool:
    """This small Code-Snipit adds an Handler to the global handlers List.

    :param handler: The Handler which will be passed on to handlers
    :type handler: logging.Handler
    :return: Return True if successfull.
    :rtype: bool
    """
    _handlers.append(handler)
    return True


def remove_handler(position: int) -> bool:
    """Removes a handler from the handler list

    :param position: _description_
    :type position: int
    :return: Return True is successfull.
    :rtype: bool
    """
    _handlers.pop(position)
    return True


def exception(
    loggername: str,
    ignore_exceptions: Optional[Union[Tuple[Type[Exception], ...], Callable[[Exception], bool]]] = None
) -> Callable:
    """
    Decorator to log exceptions in functions, with an option to ignore specified exceptions.

    This decorator logs any exceptions that occur in the decorated function using the specified logger.
    It can be applied to both synchronous and asynchronous functions.

    Note:
        Compatibility can be guaranteed only with event loops that are compatible with `asyncio.iscoroutinefunction`.

    Remember, Loggers are Singletons:
        https://docs.python.org/3/howto/logging-cookbook.html#using-loggers-as-attributes-in-a-class-or-passing-them-as-parameters

    :param loggername: The name of the logger to use for logging exceptions.
    :param ignore_exceptions: Exceptions to ignore when logging. Can be either a tuple of exception types to ignore,
                              or a callable that takes an exception instance and returns `True` if the exception
                              should be ignored. Defaults to `None`, meaning no exceptions are ignored.
    :return: The decorated function.
    """
    logger: logging.Logger = logging.getLogger(loggername)

    def log_decorator(observed_function: Callable) -> Callable:
        """
        Wraps the observed function, handling both async and non-async functions.

        :param observed_function: The function to decorate.
        :return: The wrapped function with exception logging.
        """
        if asyncio.iscoroutinefunction(observed_function):

            @wraps(observed_function)
            async def wrapper_async(*args, **kwargs):
                try:
                    return await observed_function(*args, **kwargs)
                except Exception as e:
                    should_ignore = False
                    if ignore_exceptions is not None:
                        if callable(ignore_exceptions):
                            should_ignore = ignore_exceptions(e)
                        elif isinstance(e, ignore_exceptions):
                            should_ignore = True

                    if should_ignore:
                        raise
                    error = (f"Error: exception in {observed_function.__name__}\n"
                             f"kwargs: {json.dumps(kwargs, indent=2, default=lambda o: '<not serializable>')}\n"
                             f"args: {json.dumps(args, indent=2, default=lambda o: '<not serializable>')}"
                             f"\n\n")
                    logger.exception(error)
                    raise  # re-raise the original error so the code can futher process it

            return wrapper_async

        @wraps(observed_function)
        def wrapper_line(*args, **kwargs):
            if os.environ.get('DISABLE_LOGGING_DECORATOR'):
                return observed_function(*args, **kwargs)
            try:
                return observed_function(*args, **kwargs)
            except Exception as e:
                should_ignore = False
                if ignore_exceptions is not None:
                    if callable(ignore_exceptions):
                        should_ignore = ignore_exceptions(e)
                    elif isinstance(e, ignore_exceptions):
                        should_ignore = True

                if should_ignore:
                    raise
                error = (f"Error: exception in {observed_function.__name__}\n"
                         f"kwargs: {json.dumps(kwargs, indent=2, default=lambda o: '<not serializable>')}\n"
                         f"args: {json.dumps(args, indent=2, default=lambda o: '<not serializable>')}"
                         f"\n\n")
                logger.exception(error)
                raise  # re-raise the original error so the code can futher process it

        return wrapper_line

    return log_decorator
