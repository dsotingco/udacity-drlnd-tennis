import TrajectoryBuffer
import numpy as np

def test_foo():
    assert(1+1 == 2)

def test_discount_cumsum_01():
    x = np.arange(3)
    discount = 1
    expected = np.array([3, 3, 2])
    actual = TrajectoryBuffer.discount_cumsum(x, discount)
    assert(actual[0] == expected[0])
    assert(actual[1] == expected[1])
    assert(actual[2] == expected[2])
    assert((actual == expected).all())

def test_discount_cumsum_02():
    x = np.arange(4)
    discount = 0.995
    expected = np.array([ x[0] + ( discount * x[1] ) + ( discount**2 * x[2] ) + ( discount**3 * x[3]),
                          x[1] + ( discount * x[2] ) + ( discount**2 * x[3]),
                          x[2] + ( discount * x[3] ),
                          x[3] ])
    actual = TrajectoryBuffer.discount_cumsum(x, discount)
    assert(actual[0] == expected[0])
    assert(actual[1] == expected[1])
    assert(actual[2] == expected[2])
    assert(actual[3] == expected[3])
    assert((actual == expected).all())
