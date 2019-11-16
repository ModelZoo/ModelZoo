# Copyright 2017 The Abseil Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Override DEFINE function for unnecessary help arg by Germey.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl.flags import _argument_parser, _exceptions
from absl.flags import _flag
from absl.flags import _flagvalues
from absl.flags import _helpers
from absl.flags._defines import _register_bounds_validator_if_needed


def DEFINE(parser, name, default, help=None, flag_values=_flagvalues.FLAGS, serializer=None, module_name=None, **args):
    """Registers a generic Flag object.

    NOTE: in the docstrings of all DEFINE* functions, "registers" is short
    for "creates a new flag and registers it".

    Auxiliary function: clients should use the specialized DEFINE_<type>
    function instead.

    Args:
      parser: ArgumentParser, used to parse the flag arguments.
      name: str, the flag name.
      default: The default value of the flag.
      help: str, the help message.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      serializer: ArgumentSerializer, the flag serializer instance.
      module_name: str, the name of the Python module declaring this flag.
          If not provided, it will be computed using the stack trace of this call.
      **args: dict, the extra keyword args that are passed to Flag __init__.
    """
    DEFINE_flag(_flag.Flag(parser, serializer, name, default, help, **args),
                flag_values, module_name)


def DEFINE_flag(flag, flag_values=_flagvalues.FLAGS, module_name=None):
    """Registers a 'Flag' object with a 'FlagValues' object.

    By default, the global FLAGS 'FlagValue' object is used.

    Typical users will use one of the more specialized DEFINE_xxx
    functions, such as DEFINE_string or DEFINE_integer.  But developers
    who need to create Flag objects themselves should use this function
    to register their flags.

    Args:
      flag: Flag, a flag that is key to the module.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      module_name: str, the name of the Python module declaring this flag.
          If not provided, it will be computed using the stack trace of this call.
    """
    # Copying the reference to flag_values prevents pychecker warnings.
    fv = flag_values
    fv[flag.name] = flag
    # Tell flag_values who's defining the flag.
    if module_name:
        module = sys.modules.get(module_name)
    else:
        module, module_name = _helpers.get_calling_module_object_and_name()
    flag_values.register_flag_by_module(module_name, flag)
    flag_values.register_flag_by_module_id(id(module), flag)


def DEFINE_string(name, default, help=None, flag_values=_flagvalues.FLAGS, **args):
    """Registers a flag whose value can be any string."""
    parser = _argument_parser.ArgumentParser()
    serializer = _argument_parser.ArgumentSerializer()
    DEFINE(parser, name, default, help, flag_values, serializer, **args)


def DEFINE_boolean(name, default, help=None, flag_values=_flagvalues.FLAGS, module_name=None, **args):
    """Registers a boolean flag.

    Such a boolean flag does not take an argument.  If a user wants to
    specify a false value explicitly, the long option beginning with 'no'
    must be used: i.e. --noflag

    This flag will have a value of None, True or False.  None is possible
    if default=None and the user does not specify the flag on the command
    line.

    Args:
      name: str, the flag name.
      default: bool|str|None, the default value of the flag.
      help: str, the help message.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      module_name: str, the name of the Python module declaring this flag.
          If not provided, it will be computed using the stack trace of this call.
      **args: dict, the extra keyword args that are passed to Flag __init__.
    """
    DEFINE_flag(_flag.BooleanFlag(name, default, help, **args),
                flag_values, module_name)


def DEFINE_float(name, default, help=None, lower_bound=None, upper_bound=None, flag_values=_flagvalues.FLAGS,
                 **args):  # pylint: disable=invalid-name
    """Registers a flag whose value must be a float.

    If lower_bound or upper_bound are set, then this flag must be
    within the given range.

    Args:
      name: str, the flag name.
      default: float|str|None, the default value of the flag.
      help: str, the help message.
      lower_bound: float, min value of the flag.
      upper_bound: float, max value of the flag.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      **args: dict, the extra keyword args that are passed to DEFINE.
    """
    parser = _argument_parser.FloatParser(lower_bound, upper_bound)
    serializer = _argument_parser.ArgumentSerializer()
    DEFINE(parser, name, default, help, flag_values, serializer, **args)
    _register_bounds_validator_if_needed(parser, name, flag_values=flag_values)


def DEFINE_integer(name, default, help=None, lower_bound=None, upper_bound=None, flag_values=_flagvalues.FLAGS, **args):
    """Registers a flag whose value must be an integer.

    If lower_bound, or upper_bound are set, then this flag must be
    within the given range.

    Args:
      name: str, the flag name.
      default: int|str|None, the default value of the flag.
      help: str, the help message.
      lower_bound: int, min value of the flag.
      upper_bound: int, max value of the flag.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      **args: dict, the extra keyword args that are passed to DEFINE.
    """
    parser = _argument_parser.IntegerParser(lower_bound, upper_bound)
    serializer = _argument_parser.ArgumentSerializer()
    DEFINE(parser, name, default, help, flag_values, serializer, **args)
    _register_bounds_validator_if_needed(parser, name, flag_values=flag_values)


def DEFINE_enum(name, default, enum_values, help=None, flag_values=_flagvalues.FLAGS, module_name=None, **args):
    """Registers a flag whose value can be any string from enum_values.

    Instead of a string enum, prefer `DEFINE_enum_class`, which allows
    defining enums from an `enum.Enum` class.

    Args:
      name: str, the flag name.
      default: str|None, the default value of the flag.
      enum_values: [str], a non-empty list of strings with the possible values for
          the flag.
      help: str, the help message.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      module_name: str, the name of the Python module declaring this flag.
          If not provided, it will be computed using the stack trace of this call.
      **args: dict, the extra keyword args that are passed to Flag __init__.
    """
    DEFINE_flag(_flag.EnumFlag(name, default, help, enum_values, **args),
                flag_values, module_name)


def DEFINE_enum_class(name, default, enum_class, help=None, flag_values=_flagvalues.FLAGS, module_name=None, **args):
    """Registers a flag whose value can be the name of enum members.

    Args:
      name: str, the flag name.
      default: Enum|str|None, the default value of the flag.
      enum_class: class, the Enum class with all the possible values for the flag.
      help: str, the help message.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      module_name: str, the name of the Python module declaring this flag.
          If not provided, it will be computed using the stack trace of this call.
      **args: dict, the extra keyword args that are passed to Flag __init__.
    """
    DEFINE_flag(_flag.EnumClassFlag(name, default, help, enum_class, **args),
                flag_values, module_name)


def DEFINE_list(name, default, help=None, flag_values=_flagvalues.FLAGS, **args):
    """Registers a flag whose value is a comma-separated list of strings.

    The flag value is parsed with a CSV parser.

    Args:
      name: str, the flag name.
      default: list|str|None, the default value of the flag.
      help: str, the help message.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      **args: Dictionary with extra keyword args that are passed to the
          Flag __init__.
    """
    parser = _argument_parser.ListParser()
    serializer = _argument_parser.CsvListSerializer(',')
    DEFINE(parser, name, default, help, flag_values, serializer, **args)


def DEFINE_spaceseplist(name, default, help=None, comma_compat=False, flag_values=_flagvalues.FLAGS, **args):
    """Registers a flag whose value is a whitespace-separated list of strings.

    Any whitespace can be used as a separator.

    Args:
      name: str, the flag name.
      default: list|str|None, the default value of the flag.
      help: str, the help message.
      comma_compat: bool - Whether to support comma as an additional separator.
          If false then only whitespace is supported.  This is intended only for
          backwards compatibility with flags that used to be comma-separated.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      **args: Dictionary with extra keyword args that are passed to the
          Flag __init__.
    """
    parser = _argument_parser.WhitespaceSeparatedListParser(
        comma_compat=comma_compat)
    serializer = _argument_parser.ListSerializer(' ')
    DEFINE(parser, name, default, help, flag_values, serializer, **args)


def DEFINE_multi(parser, serializer, name, default, help=None, flag_values=_flagvalues.FLAGS, module_name=None, **args):
    """Registers a generic MultiFlag that parses its args with a given parser.

    Auxiliary function.  Normal users should NOT use it directly.

    Developers who need to create their own 'Parser' classes for options
    which can appear multiple times can call this module function to
    register their flags.

    Args:
      parser: ArgumentParser, used to parse the flag arguments.
      serializer: ArgumentSerializer, the flag serializer instance.
      name: str, the flag name.
      default: Union[Iterable[T], Text, None], the default value of the flag.
          If the value is text, it will be parsed as if it was provided from
          the command line. If the value is a non-string iterable, it will be
          iterated over to create a shallow copy of the values. If it is None,
          it is left as-is.
      help: str, the help message.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      module_name: A string, the name of the Python module declaring this flag.
          If not provided, it will be computed using the stack trace of this call.
      **args: Dictionary with extra keyword args that are passed to the
          Flag __init__.
    """
    DEFINE_flag(_flag.MultiFlag(parser, serializer, name, default, help, **args),
                flag_values, module_name)


def DEFINE_multi_string(name, default, help=None, flag_values=_flagvalues.FLAGS, **args):
    """Registers a flag whose value can be a list of any strings.

    Use the flag on the command line multiple times to place multiple
    string values into the list.  The 'default' may be a single string
    (which will be converted into a single-element list) or a list of
    strings.


    Args:
      name: str, the flag name.
      default: Union[Iterable[Text], Text, None], the default value of the flag;
          see `DEFINE_multi`.
      help: str, the help message.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      **args: Dictionary with extra keyword args that are passed to the
          Flag __init__.
    """
    parser = _argument_parser.ArgumentParser()
    serializer = _argument_parser.ArgumentSerializer()
    DEFINE_multi(parser, serializer, name, default, help, flag_values, **args)


def DEFINE_multi_integer(name, default, help=None, lower_bound=None, upper_bound=None, flag_values=_flagvalues.FLAGS,
                         **args):
    """Registers a flag whose value can be a list of arbitrary integers.

    Use the flag on the command line multiple times to place multiple
    integer values into the list.  The 'default' may be a single integer
    (which will be converted into a single-element list) or a list of
    integers.

    Args:
      name: str, the flag name.
      default: Union[Iterable[int], Text, None], the default value of the flag;
          see `DEFINE_multi`.
      help: str, the help message.
      lower_bound: int, min values of the flag.
      upper_bound: int, max values of the flag.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      **args: Dictionary with extra keyword args that are passed to the
          Flag __init__.
    """
    parser = _argument_parser.IntegerParser(lower_bound, upper_bound)
    serializer = _argument_parser.ArgumentSerializer()
    DEFINE_multi(parser, serializer, name, default, help, flag_values, **args)


def DEFINE_multi_float(name, default, help=None, lower_bound=None, upper_bound=None, flag_values=_flagvalues.FLAGS,
                       **args):
    """Registers a flag whose value can be a list of arbitrary floats.

    Use the flag on the command line multiple times to place multiple
    float values into the list.  The 'default' may be a single float
    (which will be converted into a single-element list) or a list of
    floats.

    Args:
      name: str, the flag name.
      default: Union[Iterable[float], Text, None], the default value of the flag;
          see `DEFINE_multi`.
      help: str, the help message.
      lower_bound: float, min values of the flag.
      upper_bound: float, max values of the flag.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      **args: Dictionary with extra keyword args that are passed to the
          Flag __init__.
    """
    parser = _argument_parser.FloatParser(lower_bound, upper_bound)
    serializer = _argument_parser.ArgumentSerializer()
    DEFINE_multi(parser, serializer, name, default, help, flag_values, **args)


def DEFINE_multi_enum(name, default, enum_values, help=None, flag_values=_flagvalues.FLAGS, case_sensitive=True,
                      **args):
    """Registers a flag whose value can be a list strings from enum_values.

    Use the flag on the command line multiple times to place multiple
    enum values into the list.  The 'default' may be a single string
    (which will be converted into a single-element list) or a list of
    strings.

    Args:
      name: str, the flag name.
      default: Union[Iterable[Text], Text, None], the default value of the flag;
          see `DEFINE_multi`.
      enum_values: [str], a non-empty list of strings with the possible values for
          the flag.
      help: str, the help message.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      case_sensitive: Whether or not the enum is to be case-sensitive.
      **args: Dictionary with extra keyword args that are passed to the
          Flag __init__.
    """
    parser = _argument_parser.EnumParser(enum_values, case_sensitive)
    serializer = _argument_parser.ArgumentSerializer()
    DEFINE_multi(parser, serializer, name, default, help, flag_values, **args)


def DEFINE_multi_enum_class(name, default, enum_class, help=None, flag_values=_flagvalues.FLAGS, module_name=None,
                            **args):
    """Registers a flag whose value can be a list of enum members.

    Use the flag on the command line multiple times to place multiple
    enum values into the list.

    Args:
      name: str, the flag name.
      default: Union[Iterable[Enum], Iterable[Text], Enum, Text, None], the
          default value of the flag; see
          `DEFINE_multi`; only differences are documented here. If the value is
          a single Enum, it is treated as a single-item list of that Enum value.
          If it is an iterable, text values within the iterable will be converted
          to the equivalent Enum objects.
      enum_class: class, the Enum class with all the possible values for the flag.
          help: str, the help message.
      flag_values: FlagValues, the FlagValues instance with which the flag will be
        registered. This should almost never need to be overridden.
      module_name: A string, the name of the Python module declaring this flag. If
        not provided, it will be computed using the stack trace of this call.
      **args: Dictionary with extra keyword args that are passed to the Flag
        __init__.
    """
    DEFINE_flag(
        _flag.MultiEnumClassFlag(name, default, help, enum_class),
        flag_values, module_name, **args)


def DEFINE_alias(name, original_name, flag_values=_flagvalues.FLAGS, module_name=None):
    """Defines an alias flag for an existing one.

    Args:
      name: str, the flag name.
      original_name: str, the original flag name.
      flag_values: FlagValues, the FlagValues instance with which the flag will
          be registered. This should almost never need to be overridden.
      module_name: A string, the name of the module that defines this flag.

    Raises:
      flags.FlagError:
        UnrecognizedFlagError: if the referenced flag doesn't exist.
        DuplicateFlagError: if the alias name has been used by some existing flag.
    """
    if original_name not in flag_values:
        raise _exceptions.UnrecognizedFlagError(original_name)
    flag = flag_values[original_name]
    
    class _Parser(_argument_parser.ArgumentParser):
        """The parser for the alias flag calls the original flag parser."""
        
        def parse(self, argument):
            flag.parse(argument)
            return flag.value
    
    class _FlagAlias(_flag.Flag):
        """Overrides Flag class so alias value is copy of original flag value."""
        
        @property
        def value(self):
            return flag.value
        
        @value.setter
        def value(self, value):
            flag.value = value
    
    help_msg = 'Alias for --%s.' % flag.name
    # If alias_name has been used, flags.DuplicatedFlag will be raised.
    DEFINE_flag(_FlagAlias(_Parser(), flag.serializer, name, flag.default,
                           help_msg, boolean=flag.boolean),
                flag_values, module_name)


DEFINE_bool = DEFINE_boolean  # Match C++ API.

# The global FlagValues instance.
FLAGS = _flagvalues.FLAGS

define_bool = DEFINE_bool
define_boolean = define_bool
define_string = DEFINE_string
define_flag = DEFINE_flag
define_float = DEFINE_float
define_integer = DEFINE_integer
define_number = DEFINE_float
define_enum = DEFINE_enum
define_enum_class = DEFINE_enum_class
define_list = DEFINE_list
define_spaceseplist = DEFINE_spaceseplist
define_multi = DEFINE_multi
define_multi_string = DEFINE_multi_string
define_multi_integer = DEFINE_multi_integer
define_multi_float = DEFINE_multi_float
define_multi_enum = DEFINE_multi_enum
define_multi_enum_class = DEFINE_multi_enum_class


def define(name, value, help=None):
    """
    Simplify define function.
    :param name:
    :param value:
    :return:
    """
    if isinstance(value, (list, tuple)):
        define_list(name, value, help)
        return
    if isinstance(value, str):
        define_string(name, value, help)
        return
    if isinstance(value, bool):
        define_bool(name, value, help)
        return
    if isinstance(value, float):
        define_float(name, value, help)
        return
    if isinstance(value, int):
        define_integer(name, value, help)
        return
