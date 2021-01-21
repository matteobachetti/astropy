import numpy as np
import pytest
from astropy.io.ascii.qdp import read_table_qdp, write_table_qdp
from astropy.table import Table, Column
from astropy.utils.exceptions import AstropyUserWarning


def test_get_tables_from_qdp_file(tmpdir):
    example_qdp = """
    ! Swift/XRT hardness ratio of trigger: XXXX, name: BUBU X-2
    ! Columns are as labelled
    READ TERR 1
    READ SERR 2
    ! WT -- hard data
    !MJD            Err (pos)       Err(neg)        Rate            Error
    53000.123456 2.37847222222222e-05    -2.37847222222222e-05   -0.212439       0.212439
    55045.099887 1.14467592592593e-05    -1.14467592592593e-05   0.000000        0.000000
    NO NO NO NO NO
    ! WT -- soft data
    !MJD            Err (pos)       Err(neg)        Rate            Error
    53000.123456 2.37847222222222e-05    -2.37847222222222e-05   0.726155        0.583890
    55045.099887 1.14467592592593e-05    -1.14467592592593e-05   2.410935        1.393592
    NO NO NO NO NO
    ! WT -- hardness ratio
    !MJD            Err (pos)       Err(neg)        Rate            Error
    53000.123456 2.37847222222222e-05    -2.37847222222222e-05   -0.292553       -0.374935
    55045.099887 1.14467592592593e-05    -1.14467592592593e-05   0.000000        -nan
    """

    path = str(tmpdir.join('test.qdp'))

    with open(path, "w") as fp:
        print(example_qdp, file=fp)

    table0 = read_table_qdp(fp.name, input_colnames=["MJD", "Rate"], table_id=0)
    assert table0.meta["initial_comments"].lstrip().startswith("Swift")
    assert table0.meta["comments"].lstrip().startswith("WT -- hard data")
    table2 = read_table_qdp(fp.name, input_colnames=["MJD", "Rate"], table_id=2)
    assert table2.meta["initial_comments"].lstrip().startswith("Swift")
    assert table2.meta["comments"].lstrip().startswith("WT -- hardness")
    assert np.isclose(table2["MJD_nerr"][0], -2.37847222222222e-05)


def test_roundtrip(tmpdir):
    example_qdp = """
    ! Swift/XRT hardness ratio of trigger: XXXX, name: BUBU X-2
    ! Columns are as labelled
    READ TERR 1
    READ SERR 2
    ! WT -- hard data
    !MJD            Err (pos)       Err(neg)        Rate            Error
    53000.123456 2.37847222222222e-05    -2.37847222222222e-05   NO       0.212439
    55045.099887 1.14467592592593e-05    -1.14467592592593e-05   0.000000        0.000000
    NO NO NO NO NO
    ! WT -- soft data
    !MJD            Err (pos)       Err(neg)        Rate            Error
    53000.123456 2.37847222222222e-05    -2.37847222222222e-05   0.726155        0.583890
    55045.099887 1.14467592592593e-05    -1.14467592592593e-05   2.410935        1.393592
    NO NO NO NO NO
    ! WT -- hardness ratio
    !MJD            Err (pos)       Err(neg)        Rate            Error
    53000.123456 2.37847222222222e-05    -2.37847222222222e-05   -0.292553       -0.374935
    55045.099887 1.14467592592593e-05    -1.14467592592593e-05   0.000000        NO
    ! Add command, just to raise the warning.
    READ TERR 1
    ! WT -- whatever
    !MJD            Err (pos)       Err(neg)        Rate            Error
    53000.123456 2.37847222222222e-05    -2.37847222222222e-05   -0.292553       -0.374935
    NO 1.14467592592593e-05    -1.14467592592593e-05   0.000000        NO
    """

    path = str(tmpdir.join('test.qdp'))
    path2 = str(tmpdir.join('test2.qdp'))

    with open(path, "w") as fp:
        print(example_qdp, file=fp)
    with pytest.warns(AstropyUserWarning) as record:
        table = read_table_qdp(fp.name, input_colnames=["MJD", "Rate"],
                               table_id=0)
    assert np.any(["This file contains multiple command blocks"
                   in r.message.args[0]
                   for r in record])

    write_table_qdp(table, path2)

    new_table = read_table_qdp(path2, input_colnames=["MJD", "Rate"], table_id=0)

    for col in new_table.colnames:
        is_masked = np.array([np.ma.is_masked(val) for val in new_table[col]])
        if np.any(is_masked):
            # All NaN values are read as such.
            assert np.ma.is_masked(table[col][is_masked])

        is_nan = np.array([(not np.ma.is_masked(val) and np.isnan(val))
                           for val in new_table[col]])
        # All non-NaN values are the same
        assert np.allclose(new_table[col][~is_nan], table[col][~is_nan])
        if np.any(is_nan):
            # All NaN values are read as such.
            assert np.isnan(table[col][is_nan])
    assert np.allclose(new_table['MJD_perr'], [2.378472e-05, 1.1446759e-05])

    for meta_name in ['initial_comments', 'comments']:
        assert meta_name in new_table.meta


def test_read_write_simple(tmpdir):
    test_file = str(tmpdir.join('test.qdp'))
    t1 = Table()
    t1.add_column(Column(name='a', data=[1, 2, 3]))
    t1.write(test_file, format='qdp')
    with pytest.warns(UserWarning) as record:
        t2 = Table.read(test_file, format='qdp')
    assert np.any(["table_id not specified. Reading the first available table"
                   in r.message.args[0]
                   for r in record])
    assert np.all(t2['col1'] == t1['a'])


def test_read_write_simple_specify_name(tmpdir):
    test_file = str(tmpdir.join('test.qdp'))
    t1 = Table()
    t1.add_column(Column(name='a', data=[1, 2, 3]))
    # Give a non-None err_specs
    t1.write(test_file, format='qdp', err_specs={})
    t2 = Table.read(test_file, table_id=0, format='qdp', input_colnames=['a'])
    assert np.all(t2['a'] == t1['a'])
