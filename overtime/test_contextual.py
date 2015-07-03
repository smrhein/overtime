import unittest

from contextual import get_context_guard


class Test_get_context_guard(unittest.TestCase):
    def test_call(self):
        with open('test.txt', 'r+') as f:
            enter, exit_ = get_context_guard(f)
            self.assertEqual(enter, f.__enter__)
            self.assertEqual(exit_, f.__exit__)

        self.assertRaises(TypeError, get_context_guard, '')


class Test_Exiting(unittest.TestCase):
    def test_call(self):
        raise NotImplementedError


class Test_Iterated(unittest.TestCase):
    def test_call(self):
        raise NotImplementedError


class Test_GenericContextManager(unittest.TestCase):
    def test_call(self):
        raise NotImplementedError


class Test_ctx_delayed(unittest.TestCase):
    def test_call(self):
        raise NotImplementedError


class Test_ctx_worker(unittest.TestCase):
    def test_call(self):
        raise NotImplementedError


class Test_CtxObject(unittest.TestCase):
    def test_call(self):
        raise NotImplementedError


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
