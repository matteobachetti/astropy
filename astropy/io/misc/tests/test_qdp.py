import numpy as np
from astropy.io.misc.qdp import read_table_qdp, write_table_qdp


def test_get_tables_from_qdp_file():
    import tempfile

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

    fd, path = tempfile.mkstemp()
    with open(path, "w") as fp:
        print(example_qdp, file=fp)

    table0 = read_table_qdp(fp.name, input_colnames=["MJD", "Rate"], table_id=0)
    assert table0.meta["initial_comments"].lstrip().startswith("Swift")
    assert table0.meta["comments"].lstrip().startswith("WT -- hard data")
    table2 = read_table_qdp(fp.name, input_colnames=["MJD", "Rate"], table_id=2)
    assert table2.meta["initial_comments"].lstrip().startswith("Swift")
    assert table2.meta["comments"].lstrip().startswith("WT -- hardness")
    assert np.isclose(table2["MJD_nerr"][0], -2.37847222222222e-05)


def test_roundtrip():
    import tempfile

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
    """

    fd, path = tempfile.mkstemp()
    fd2, path2 = tempfile.mkstemp()
    with open(path, "w") as fp:
        print(example_qdp, file=fp)

    table = read_table_qdp(fp.name, input_colnames=["MJD", "Rate"], table_id=0)

    write_table_qdp(table, path2)

    new_table = read_table_qdp(path2, input_colnames=["MJD", "Rate"], table_id=0)

    assert np.allclose(new_table['MJD'], table['MJD'])
    assert np.isnan(table['Rate'][0])
    assert np.isnan(new_table['Rate'][0])

    for meta_name in ['initial_comments', 'comments']:
        assert meta_name in new_table.meta
