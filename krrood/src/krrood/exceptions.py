from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional


@dataclass
class DataclassException(Exception):
    """
    A base exception class for dataclass-based exceptions.
    The way this is used is by inheriting from it and setting the `message` field in the __post_init__ method,
    then calling the super().__post_init__() method.
    """

    message: str = field(kw_only=True, default=None)

    def __post_init__(self):
        super().__init__(self.message)


@dataclass
class InputError(DataclassException):
    """
    Raised when there is an error with user input.
    """


@dataclass
class ModuleNotFoundForConvertingImportsToAbsolute(InputError):
    """
    Raised when the current module is not given and/or not found for relative import conversion.
    """
    path: Optional[str] = field(kw_only=True, default=None)
    """
    The path to the file that contains the relative import.
    """
    source_code: Optional[str] = field(kw_only=True, default=None)
    """
    The source code of the file that contains the relative import.
    """

    def __post_init__(self):
        self.message = (f"Current module is required for relative import conversion, path: {self.path},"
                        f" source code: {self.source_code}.")
        super().__post_init__()


@dataclass
class NoSourceDataToParseImportsFrom(InputError):
    """
    Raised when there is no source data given to parse imports from.
    """


@dataclass
class NoModuleSourceProvided(InputError):
    """
    Raised when there is no source module data given to parse imports from.
    """


@dataclass
class NoDefaultValueFound(DataclassException):
    """
    Raised when no default value for a given field in a dataclass is found.
    """


@dataclass
class PackageNameNotFoundError(DataclassException):
    """
    Raised when a package name is not found in a given path.
    """


@dataclass
class PathMissingRequiredComponentsError(DataclassException):
    """
    Raised when a path does not contain all required components.
    """


@dataclass
class SubprocessExecutionError(DataclassException):
    """
    Raised when a subprocess execution fails.
    """


@dataclass
class SourceDataNotProvided(InputError):
    """
    Raised when no source data is provided.
    """
