"""
Base chain handler classes for Chain of Responsibility pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class Handler(ABC):
    """Abstract base class for handlers in Chain of Responsibility pattern."""

    def __init__(self) -> None:
        self._next_handler: Optional['Handler'] = None

    def set_next(self, handler: 'Handler') -> 'Handler':
        """Set next handler in chain.

        Args:
            handler: Next handler to process data

        Returns:
            The handler that was set (for chaining)
        """
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, data: Any) -> Any:
        """Process data. Must be implemented by subclasses.

        Args:
            data: Input data to process

        Returns:
            Processed data
        """
        pass

    def _next(self, data: Any) -> Any:
        """Pass data to next handler in chain.

        Args:
            data: Data to pass to next handler

        Returns:
            Result from next handler or original data if no next handler
        """
        if self._next_handler:
            return self._next_handler.handle(data)
        return data


class Pipeline:
    """Pipeline class placeholder."""
    pass
