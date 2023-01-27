SUBSET_LENGTHS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def test_pytest_working():
    assert 1 == 1


def test_if_count_approx_50000():
    """
    Authors of Quantifying Memorization Across Neural Language Models claim that they get approximately 50,000
    individual duplicated samples per length. This test for 40,000.
    """
    for length in SUBSET_LENGTHS:
        # open something
        # assert len(something = length
        pass
