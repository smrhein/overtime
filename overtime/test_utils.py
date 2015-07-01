import unittest

from overtime.utils import CallBuilder, unpack_nfirst


class Test_unpack_nfirst(unittest.TestCase):
    def test_call(self):
        with self.assertRaises(ValueError):
            first, rest = unpack_nfirst((), 1)

        with self.assertRaises(ValueError):
            first, second, rest = unpack_nfirst((0,), 2)

        first, rest = unpack_nfirst((0, 1), 1)
        self.assertEqual(first, 0)
        self.assertEqual(rest, (1,))

        first, second, rest = unpack_nfirst((0, 1, 2, 3), 2)
        self.assertEqual(first, 0)
        self.assertEqual(second, 1)
        self.assertEqual(rest, (2, 3))


class Test_CallBuilder(unittest.TestCase):
    def test_extend(self):
        v = list((0, 1, 2, 3))
        c = CallBuilder(v, 'extend')
        c = c.extend_args([v])
        c()
        self.assertEqual(v, [0, 1, 2, 3] * 2)

        v = list((0, 1, 2, 3))
        c = CallBuilder(list, 'extend')
        c = c.extend_args([v, v])
        c()
        self.assertEqual(v, [0, 1, 2, 3] * 2)

    def test_update(self):
        v = {'a': 0, 'b': 1, 'c': 2}
        c = CallBuilder(v, 'update')
        c = c.update_kwargs(a=2, b=1, c=0)
        c()
        self.assertEqual(v, {'a': 2, 'b': 1, 'c': 0})

        v = {'a': 0, 'b': 1, 'c': 2}
        c = CallBuilder(dict, 'update')
        c = c.extend_args([v])
        c = c.update_kwargs(a=2, b=1, c=0)
        c()
        self.assertEqual(v, {'a': 2, 'b': 1, 'c': 0})

    def test_call(self):
        fmt = '{0} {1} {2} {zero} {one} {two}'
        c = CallBuilder(fmt, 'format')
        c = c.extend_args(['zero', 'one', 'two'])
        c = c.update_kwargs(zero=0, one=1, two=2)
        self.assertEqual(c(), 'zero one two 0 1 2')


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
