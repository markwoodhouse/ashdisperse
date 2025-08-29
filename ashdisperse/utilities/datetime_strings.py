import datetime
import re


def is_valid_datetime(input_datetime):
    datetime_fmt = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}')
    return datetime_fmt.match(input_datetime) is not None


def string_to_datetime(datetime_string):
    if not is_valid_datetime(datetime_string):
        raise ValueError('{} is not a valid datetime string.  Required format is yyyy-mm-dd hh:mm'.format(datetime_string))
    (date, time) = datetime_string.split(' ', 2)
    [year, month, day] = [int(x) for x in date.split('-', 3)]
    [hour, minute] = [int(x) for x in time.split(':', 2)]

    output_datetime = (datetime.datetime(year=year, month=month, day=day,
                                             hour=hour, minute=0)
                            + datetime.timedelta(hours=minute//30))

    return output_datetime