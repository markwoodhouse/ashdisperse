import datetime
import re
from os.path import isfile

import pandas as pd
from numpy import isfinite
from prompt_toolkit import HTML, print_formatted_text, prompt
from prompt_toolkit.completion import PathCompleter, WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import ValidationError, Validator

from ashdisperse.data import get_gvp_list
from ashdisperse.utilities import is_valid_datetime, string_to_datetime



def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            ret = True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            ret = False  # Terminal running IPython
        else:
            ret = False  # Other type (?)
        return ret
    except NameError:
        return False  # Probably standard Python interpreter


__ISNOTEBOOK = isnotebook()

style = Style.from_dict(
    {
        "title": "#3b76fe bold",
        "text": "#3b76fe",
        "query": "#33cc33",
        "query_ans": "#33cc33 italic",
        "warning": "#e20588",
        "error": "#ff0000",
        "fatal_error": "#ff0000 bold",
    }
)

gvp_list = get_gvp_list()

def print_title(title_text):
    if __ISNOTEBOOK:
        print(title_text)
    else:
        print_formatted_text(
            HTML("<title>{}\n</title>".format(title_text)), style=style
        )


def print_text(text):
    if __ISNOTEBOOK:
        print(text)
    else:
        print_formatted_text(HTML("<text> {} </text>".format(text)), style=style)


def print_warning(text):
    if __ISNOTEBOOK:
        print(text)
    else:
        print_formatted_text(HTML("<warning> {} </warning>".format(text)), style=style)

def query_yes_no(question, default="yes"):

    if default is None:
        ans = " [y/n] "
    elif default == "yes":
        ans = " [Y/n] "
    elif default == "no":
        ans = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}

    if __ISNOTEBOOK:
        question_text = "{} {}:".format(question, ans)
        error_text = "Please respond with 'yes' or 'no' (or 'y' or 'n')"
    else:
        question_text = HTML(
            "<query> {} </query> <query_ans>".format(question)
            + "{}: </query_ans>".format(ans)
        )
        error_text = HTML(
            "<error> Please respond with 'yes' or 'no'" + "(or 'y' or 'n').\n </error>"
        )

    while True:
        if __ISNOTEBOOK:
            choice = input(question_text).lower()
        else:
            choice = prompt(question_text, style=style).lower()

        if default is not None and choice == "":
            return valid[default]
        if choice in valid:
            return valid[choice]

        if __ISNOTEBOOK:
            print(error_text)
        else:
            print_formatted_text(error_text, style=style)


def query_choices(question, choices=["y", "n"], default=None):
    """Asks a question via input() with answer in choices.

    Args:
        question: a string that is presented to the user.
        choices: a list of allowed responses
        **default: Optional; the presumed answer if the user just hits <Enter>.

    Returns:
        The "answer" return value is True for "yes" or False for "no".
    """
    ans = "[" + "/".join(choices) + "]"
    if default is not None:
        ans += " (default {})".format(default)

    if __ISNOTEBOOK:
        question_text = "{} {}: ".format(question, ans)
        error_text = "Please respond with one of " + ",".join(choices) + "."
    else:
        question_text = HTML(
            "<query> {} </query> <query_ans>".format(question)
            + "{}: </query_ans>".format(ans)
        )
        error_text = HTML(
            "<error> Please respond with one of " + ",".join(choices) + ".\n </error>"
        )
        ans_completer = WordCompleter(choices)

    while True:
        if __ISNOTEBOOK:
            choice = input(question_text)
        else:
            choice = prompt(
                question_text,
                style=style,
                completer=ans_completer,
                complete_while_typing=True,
            )

        if default is not None and choice == "":
            return default
        if choice in choices:
            return choice
        if __ISNOTEBOOK:
            print(error_text)
        else:
            print_formatted_text(error_text, style=style)


def query_change_value(question, default=None, lower=None, upper=None, answer_type=float):
    """Ask a question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be a value
    "answer_type" is the type of the answer expected.

    """

    if default is not None and not isinstance(default, answer_type):
        raise ValueError(
            "In query_change_value, default value {}".format(default)
            + "must be of type {}".format(answer_type)
        )
    
    if lower is not None and not isinstance(lower, answer_type):
        raise ValueError(
            f"In query_change_value, lower value {lower} must be of type {answer_type}"
        )

    if default is None:
        ans = " "
    else:
        ans = " [return for default = {}] ".format(default)

    def is_valid_num(text):

        if default is not None and len(text)==0:
            return True

        try:
            answer_type(text)
        except ValueError:
            return False

        val = answer_type(text)
        if lower is not None:
            if answer_type(text)<lower:
                return False
        if upper is not None:
            if answer_type(text)>upper:
                return False
        
        return True

    error_message="This input contains non-numeric characters"
    if lower is not None and upper is None:
        error_message += f" or is <{lower}."
    if lower is None and upper is not None:
        error_message += f" or is >{upper}."
    if lower is not None and upper is not None:
        error_message += f" or is  not in the range {lower} -- {upper}."
    if __ISNOTEBOOK:
        question_text = "{} {}:".format(question, ans)
    else:
        validator = Validator.from_callable(
            is_valid_num,
            error_message=error_message,
            move_cursor_to_end=True,
        )
        question_text = HTML(
            "<query> {} </query> <query_ans>".format(question)
            + " {}: </query_ans>".format(ans)
        )

    while True:
        if __ISNOTEBOOK:
            choice = input(question_text)
        else:
            choice = prompt(question_text, style=style, validator=validator)

        if default is not None and choice == "":
            ret = default
        else:
            ret = answer_type(choice)
        return ret


def query_set_value(question, answer_type=float, lower=None, upper=None):
    """Ask a question via input() and return their answer.

    "question" is a string that is presented to the user.
    "answer_type" is the type of the answer expected.

    """

    def is_valid_num(text):
        try:
            val = answer_type(text)
            if lower is not None:
                if val<lower:
                    return False
            if upper is not None:
                if val>upper:
                    return False
            return True
        except ValueError:
            return False

    error_message="This input contains non-numeric characters"
    if lower is not None and upper is None:
        error_message += f" or is <{lower}."
    if lower is None and upper is not None:
        error_message += f" or is >{upper}."
    if lower is not None and upper is not None:
        error_message += f" or is  not in the range {lower} -- {upper}."
    if __ISNOTEBOOK:
        question_text = "{} : ".format(question)
    else:
        validator = Validator.from_callable(
            is_valid_num,
            error_message=error_message,
            move_cursor_to_end=True,
        )
        question_text = HTML("<query> {} </query> : ".format(question))

    while True:
        if __ISNOTEBOOK:
            choice = input(question_text)
        else:
            choice = prompt(question_text, style=style, validator=validator)

        return answer_type(choice)


def query_datetime(question):
    if __ISNOTEBOOK:
        question_text = "{} :".format(question)
    else:
        validator = Validator.from_callable(
            is_valid_datetime,
            error_message="This input is not recognized as a datetime string",
            move_cursor_to_end=True,
        )
        question_text = HTML("<query> {} </query> : ".format(question))

    while True:
        if __ISNOTEBOOK:
            input_datetime = input(question_text)
        else:
            input_datetime = prompt(question_text, style=style, validator=validator)

        output_datetime = string_to_datetime(input_datetime)

        return output_datetime


def is_valid_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def is_valid_lat(lat):
    lat = float(lat)
    if lat > 90 or lat < -90:
        ret = False
    else:
        ret = True
    return ret


def is_valid_lon(lon):
    lon = float(lon)
    if lon > 360 or lon < -180:
        ret = False
    else:
        ret = True
    return ret


def is_valid_latlon(latlon):
    try:
        lat, lon = latlon.split(",", 2)
        lat = float(lat)
        lon = float(lon)
        if is_valid_lat(lat) & is_valid_lon(lon):
            ret = (lat, lon)
        else:
            ret = (None, None)
        return ret
    except ValueError:
        return (None, None)


def has_comma(text):
    text_split = text.split(",")
    if len(text_split) != 2:
        ret = False
    else:
        ret = True
    return ret


def get_latlon_from_gvp_name(name):
    this_volc = gvp_list.loc[gvp_list.Search_Name == name]
    indx = this_volc['Primary_indx']

    lat = gvp_list.loc[indx, "Latitude"].values[0]
    lon = gvp_list.loc[indx, "Longitude"].values[0]

    return name, lat, lon


class NameLatLonValidator(Validator):
    def validate(self, document):
        text = document.text

        # if not has_comma(text):

        if has_comma(text) and (
            text[0].isnumeric() or (text[0] == "-" and text[1].isnumeric())
        ):
            lat_i = text.find(",")
            lat, lon = text.split(",", 2)
            if not is_valid_float(lat):
                raise ValidationError(
                    message="Specify latitude in decimal degrees", cursor_position=lat_i
                )
            if not is_valid_lat(lat):
                raise ValidationError(
                    message="Specify latitude in decimal degrees in "
                    + "range -90 -- 90",
                    cursor_position=lat_i,
                )
            if not is_valid_float(lon):
                raise ValidationError(
                    message="Specify longitude in decimal degrees",
                    cursor_position=len(text),
                )
            if not is_valid_lon(lon):
                raise ValidationError(
                    message="Specify latitude in decimal degrees "
                    + "in range -180--180 or 0--360",
                    cursor_position=len(text),
                )
        else:
            if text not in gvp_list.Search_Name.values:
                raise ValidationError(
                    message="Specify location as volcano name"
                    + " or as latitude and longitude "
                    + "(as decimal degrees in format lat, lon)",
                    cursor_position=len(text),
                )


def query_latlon():
    latlon_query_text = (
        "Enter Volcano name or give latitude and "
        + "longitude of source (as decimal degrees in format lat, lon): "
    )

    if __ISNOTEBOOK:
        while True:
            latlon = input(latlon_query_text)
            lat, lon = is_valid_latlon(latlon)
            if lat is not None and lon is not None:
                name = None
                break

            if latlon in gvp_list.Search_Name.values:
                name, lat, lon = get_latlon_from_gvp_name(latlon)
                break

            print(
                "Not a recognized volcano name nor a valid "
                + "latitude-longitude coordinate"
            )
            print(
                "Specify latitude and longitude "
                + "(as decimal degrees in format lat, lon)"
            )
    else:
        latlon_query = HTML("<query> {} </query>".format(latlon_query_text))
        name_cmplter = WordCompleter(gvp_list.Search_Name.unique())
        latlon = prompt(
            latlon_query,
            completer=name_cmplter,
            validator=NameLatLonValidator(),
            style=style,
        )
        if has_comma(latlon) and latlon[0].isnumeric():
            lat, lon = is_valid_latlon(latlon)
            name = None
        else:
            name, lat, lon = get_latlon_from_gvp_name(latlon)

    print(name, lat, lon)

    return name, lat, lon


def query_met_file():
    met_query = "Path of met file (netCDF): "
    if __ISNOTEBOOK:
        while True:
            met_file = input(met_query)
            if not isfile(met_file):
                print_text("Met file {} does not exist".format(met_file))
            else:
                break
    else:
        met_file = prompt(met_query, completer=PathCompleter(expanduser=True))
    return met_file
