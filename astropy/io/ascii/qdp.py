# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains functions for reading and writing QDP tables that are
not meant to be used directly, but instead are available as readers/writers in
`astropy.table`. See :ref:`table_io` for more details.
"""
import re
import copy
from itertools import groupby
import numpy as np
import warnings
from astropy.utils.exceptions import AstropyUserWarning
from astropy.table import Table
from astropy.io import registry as io_registry

from . import core, basic


def is_qdp(origin, filepath, fileobj, *args, **kwargs):
    if filepath is not None:
        return filepath.endswith(('.qdp'))
    return False


_decimal_re = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
_command_re = r'READ [TS]ERR(\s+[0-9]+)+'
_new_re = r'NO(\s+NO)+'
_data_re = rf'({_decimal_re}|NO|[-+]?nan)(\s+({_decimal_re}|NO|[-+]?nan))*)'
_line_type_re = re.compile(rf'^\s*((?P<command>{_command_re})|(?P<new>{_new_re})|(?P<data>{_data_re})?\s*(\!(?P<comment>.*))?\s*$')


def line_type(line):
    """Interpret a QDP file line

    Parameters
    ----------
    line : str
        a single line of the file

    Returns
    -------
    type : str
        Line type: "comment", "command", or "data"

    Examples
    --------
    >>> line_type("READ SERR 3")
    'command'
    >>> line_type(" \\n    !some gibberish")
    'comment'
    >>> line_type("   ")
    'comment'
    >>> line_type(" 21345.45")
    'data,1'
    >>> line_type(" 21345.45 1.53e-3 1e-3 .04 NO nan")
    'data,6'
    >>> line_type(" 21345.45 ! a comment to disturb")
    'data,1'
    >>> line_type("NO NO NO NO NO")
    'new'
    >>> line_type("N O N NOON OON O")
    Traceback (most recent call last):
        ...
    ValueError: Unrecognized QDP line...
    >>> line_type(" some non-comment gibberish")
    Traceback (most recent call last):
        ...
    ValueError: Unrecognized QDP line...
    """
    line = line.strip()
    if not line:
        return 'comment'
    match = _line_type_re.match(line)
    if match is None:
        raise ValueError(f'Unrecognized QDP line: {line}')
    for type_, val in match.groupdict().items():
        if val is None:
            continue

        if type_ == 'data':
            return f'data,{len(val.split())}'
        else:
            return type_


def get_type_from_list_of_lines(lines):
    """Read through the list of QDP file lines and label each line by type

    Parameters
    ----------
    lines : list
        List containing one file line in each entry

    Returns
    -------
    contents : list
        List containing the type for each line (see `line_type_and_data`)
    ncol : int
        The number of columns in the data lines. Must be the same throughout
        the file

    Examples
    --------
    >>> line0 = "! A comment"
    >>> line1 = "543 12 456.0"
    >>> lines = [line0, line1]
    >>> types, ncol = get_type_from_list_of_lines(lines)
    >>> types[0]
    'comment'
    >>> types[1]
    'data,3'
    >>> ncol
    3
    >>> lines.append("23")
    >>> get_type_from_list_of_lines(lines)
    Traceback (most recent call last):
        ...
    ValueError: Inconsistent number of columns
    """

    types = [line_type(line) for line in lines]
    current_ncol = None
    for type_ in types:
        if type_.startswith('data', ):
            ncol = int(type_[5:])
            if current_ncol is None:
                current_ncol = ncol
            elif ncol != current_ncol:
                raise ValueError('Inconsistent number of columns')

    return types, current_ncol


def analyze_qdp_file(qdp_file):
    """Read through the QDP file and label each line by type

    Parameters
    ----------
    qdp_file : str
        File name

    Returns
    -------
    contents : list
        List containing the type for each line (see `line_type_and_data`)
    ncol : int
        The number of columns in the data lines. Must be the same throughout
        the file
    """
    with open(qdp_file) as fobj:
        lines = fobj.readlines()

    return get_type_from_list_of_lines(lines)


def interpret_err_lines(err_specs, ncols, input_colnames=None):
    """Give list of column names from the READ SERR and TERR commands

    Parameters
    ----------
    err_specs : dict, {'serr': [n0, n1, ...], 'terr': [n2, n3, ...]}
        Error specifications for symmetric and two-sided errors
    ncols : int
        Number of data columns

    Other parameters
    ----------------
    input_colnames : list of strings
        Name of data columns (defaults to ['col1', 'col2', ...]), _not_
        including error columns.

    Returns
    -------
    colnames : list
        List containing the column names. Error columns will have the name
        of the main column plus ``_err`` for symmetric errors, and ``_perr``
        and ``_nerr`` for positive and negative errors respectively

    Examples
    --------
    >>> col_in = ['MJD', 'Rate']
    >>> cols = interpret_err_lines(None, 2, input_colnames=col_in)
    >>> cols[0]
    'MJD'
    >>> err_specs = {'terr': [1], 'serr': [2]}
    >>> ncols = 5
    >>> cols = interpret_err_lines(err_specs, ncols, input_colnames=col_in)
    >>> cols[0]
    'MJD'
    >>> cols[2]
    'MJD_nerr'
    >>> cols[4]
    'Rate_err'
    >>> interpret_err_lines(err_specs, 6, input_colnames=col_in)
    Traceback (most recent call last):
        ...
    ValueError: Inconsistent number of input colnames
    """

    colnames = ["" for i in range(ncols)]
    if err_specs is None:
        serr_cols = terr_cols = []

    else:
        # I don't want to empty the original one when using `pop` below
        err_specs = copy.deepcopy(err_specs)

        serr_cols = err_specs.pop("serr", [])
        terr_cols = err_specs.pop("terr", [])

    if input_colnames is not None:
        all_error_cols = len(serr_cols) + len(terr_cols) * 2
        if all_error_cols + len(input_colnames) != ncols:
            raise ValueError("Inconsistent number of input colnames")

    shift = 0
    for i in range(ncols):
        col_num = i + 1 - shift
        if colnames[i] != "":
            continue

        colname_root = f"col{col_num}"

        if input_colnames is not None:
            colname_root = input_colnames[col_num - 1]

        colnames[i] = f"{colname_root}"
        if col_num in serr_cols:
            colnames[i + 1] = f"{colname_root}_err"
            shift += 1
            continue

        if col_num in terr_cols:
            colnames[i + 1] = f"{colname_root}_perr"
            colnames[i + 2] = f"{colname_root}_nerr"
            shift += 2
            continue

    assert not np.any([c == "" for c in colnames])

    return colnames


def get_tables_from_qdp_file(qdp_file, input_colnames=None):
    """Get all tables from a QDP file

    Parameters
    ----------
    qdp_file : str
        Input QDP file name

    Other parameters
    ----------------
    input_colnames : list of strings
        Name of data columns (defaults to ['col1', 'col2', ...]), _not_
        including error columns.

    Returns
    -------
    tables : list of `Table` objects
        List containing all the tables present inside the QDP file
    """

    contents, ncol = analyze_qdp_file(qdp_file)

    with open(qdp_file) as fobj:
        lines = fobj.readlines()

    table_list = []
    err_specs = {}
    colnames = None

    comment_text = ""
    initial_comments = ""
    command_lines = ""
    current_rows = None

    for line, datatype in zip(lines, contents):
        line = line.strip().lstrip('!')
        # Is this a comment?
        if datatype == "comment":
            comment_text += line + '\n'
            continue

        if datatype == "command":
            # The first time I find commands, I save whatever comments into
            # The initial comments.
            if command_lines == "":
                initial_comments = comment_text
                comment_text = ""

            if err_specs != {}:
                warnings.warn(
                    "This file contains multiple command blocks. Please verify",
                    AstropyUserWarning
                )
            command_lines += line + '\n'
            continue

        if datatype.startswith("data"):
            # The first time I find data, I define err_specs
            if err_specs == {} and command_lines != "":
                for cline in command_lines.strip().split('\n'):
                    command = cline.strip().split()
                    if len(command) < 3:
                        continue
                    err_specs[command[1].lower()] = [int(c) for c in
                                                     command[2:]]
            if colnames is None:
                colnames = interpret_err_lines(
                    err_specs, ncol, input_colnames=input_colnames
                )

            if current_rows is None:
                current_rows = []

            values = []
            for v in line.split():
                if v == "NO":
                    values.append(np.ma.masked)
                else:
                    values.append(float(v))
            current_rows.append(values)
            continue
        if datatype == "new":
            # Save table to table_list and reset
            if current_rows is not None:
                new_table = Table(names=colnames, rows=current_rows)
                new_table.meta["initial_comments"] = initial_comments.strip()
                new_table.meta["comments"] = comment_text.strip()
                # Reset comments
                comment_text = ""
                table_list.append(new_table)
                current_rows = None
            continue
    # At the very end, if there is still a table being written, let's save
    # it to the table_list
    if current_rows is not None:
        new_table = Table(names=colnames, rows=current_rows)
        new_table.meta["initial_comments"] = initial_comments.strip()
        new_table.meta["comments"] = comment_text.strip()
        table_list.append(new_table)

    return table_list


def understand_err_col(colnames):
    """Get which column names are error columns

    Examples
    --------
    >>> colnames = ['a', 'a_err', 'b', 'b_perr', 'b_nerr']
    >>> serr, terr = understand_err_col(colnames)
    >>> np.allclose(serr, [1])
    True
    >>> np.allclose(terr, [2])
    True
    >>> serr, terr = understand_err_col(['a', 'a_nerr'])
    Traceback (most recent call last):
    ...
    ValueError: Missing positive error...
    >>> serr, terr = understand_err_col(['a', 'a_perr'])
    Traceback (most recent call last):
    ...
    ValueError: Missing negative error...
    """
    shift = 0
    serr = []
    terr = []

    for i, col in enumerate(colnames):
        if col.endswith("_err"):
            # The previous column, but they're numbered from 1!
            # Plus, take shift into account
            serr.append(i - shift)
            shift += 1
        elif col.endswith("_perr"):
            terr.append(i - shift)
            if len(colnames) == i + 1 or not colnames[i + 1].endswith('_nerr'):
                raise ValueError("Missing negative error")
            shift += 2
        elif col.endswith("_nerr") and not colnames[i - 1].endswith('_perr'):
            raise ValueError("Missing positive error")
    return serr, terr


def read_table_qdp(qdp_file, input_colnames=None, table_id=None):
    """Read a table from a QDP file

    Parameters
    ----------
    qdp_file : str
        Input QDP file name

    Other parameters
    ----------------
    table_id : int, default 0
        Number of the table to be read from the QDP file. This is useful
        when multiple tables present in the file. By default, the first is read.

    input_colnames : list of strings
        Name of data columns (defaults to ['col1', 'col2', ...]), _not_
        including error columns.

    Returns
    -------
    tables : list of `Table` objects
        List containing all the tables present inside the QDP file
    """
    if table_id is None:
        warnings.warn("table_id not specified. Reading the first available "
                      "table", AstropyUserWarning)
        table_id = 0

    tables = get_tables_from_qdp_file(qdp_file, input_colnames=input_colnames)

    return tables[table_id]


def write_table_qdp(table, filename, err_specs=None):
    """Write a table to a QDP file

    Parameters
    ----------
    table : :class:`~astropy.table.Table` object
        Input table to be written
    filename : str
        Output QDP file name

    Other parameters
    ----------------
    err_specs : dict
        Dictionary of the format {'serr': [1], 'terr': [2, 3]}, specifying
        which columns have symmetric and two-sided errors (see QDP format
        specification)
    """
    from astropy.io import ascii
    with open(filename, 'w') as fobj:
        if 'initial_comments' in table.meta and table.meta['initial_comments'].strip() != "":
            for line in table.meta['initial_comments'].split("\n"):
                line = line.strip()
                if not line.startswith("!"):
                    line = "!" + line
                print(line, file=fobj)

        if err_specs is None:
            serr_cols, terr_cols = understand_err_col(table.colnames)
        else:
            serr_cols = err_specs.pop("serr", [])
            terr_cols = err_specs.pop("terr", [])
        if serr_cols != []:
            col_string = " ".join([str(val) for val in serr_cols])
            print(f"READ SERR {col_string}", file=fobj)
        if terr_cols != []:
            col_string = " ".join([str(val) for val in terr_cols])
            print(f"READ TERR {col_string}", file=fobj)

        if 'comments' in table.meta and table.meta['comments'].strip() != "":
            for line in table.meta['comments'].split("\n"):
                line = line.strip()
                if not line.startswith("!"):
                    line = "!" + line
                print(line, file=fobj)

        colnames = table.colnames
        print("!" + " ".join(colnames), file=fobj)
        for row in table:
            values = []
            for val in row:
                if not np.ma.is_masked(val):
                    rep = str(val)
                else:
                    rep = "NO"
                values.append(rep)
            print(" ".join(values), file=fobj)


class QDPSplitter(core.DefaultSplitter):
    """
    Split on comma for QDP tables
    """
    delimiter = ' '


class QDPHeader(basic.BasicHeader):
    """
    Header that uses the :class:`astropy.io.ascii.basic.QDPSplitter`
    """
    splitter_class = QDPSplitter
    comment = None
    write_comment = None


class QDPData(basic.BasicData):
    """
    Data that uses the :class:`astropy.io.ascii.basic.CsvSplitter`
    """
    splitter_class = QDPSplitter
    fill_values = [(core.masked, '')]
    comment = None
    write_comment = None


class QDP(basic.Basic):
    """QDP table.

    This file format can contain multiple tables, separated by a line full
    of ``NO``s. Comments are exclamation marks, and missing values are single
    ``NO`` entries.
    Headers are just comments, and tables distributed by various missions
    can differ greatly in their use of conventions. For examples, light curves
    distributed by the Swift-Gehrels mission have an extra space in one header
    entry that makes the number of  inconsistent with the number of columns

    ..
                      Extra space
                          |
                          v
       !     MJD       Err (pos)       Err(neg)        Rate            Error
        53000.123456   2.378e-05     -2.378472e-05     NO             0.212439


    """
    _format_name = 'qdp'
    _io_registry_format_aliases = ['qdp']
    _io_registry_can_write = True
    _io_registry_suffix = '.qdp'
    _description = 'Quick and Dandy Plotter'

    header_class = QDPHeader
    data_class = QDPData


def register_qdp():
    io_registry.register_reader('qdp', Table, read_table_qdp)
    io_registry.register_writer('qdp', Table, write_table_qdp)
    io_registry.register_identifier('qdp', Table, is_qdp)