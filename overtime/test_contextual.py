import unittest
from contextual import  get_context_guard, contextual_task, IteratedContextManager, GenericContextManager


class Test_get_context_guard(unittest.TestCase):
    def test_call(self):
        with open('test.txt', 'r+') as f:
            enter, exit_ = get_context_guard(f)
            self.assertEqual(enter, f.__enter__)
            self.assertEqual(exit_, f.__exit__)
            
        self.assertRaises(TypeError, get_context_guard, '')
        

class Test_contextual_task(unittest.TestCase):
    def test_call(self):
        self.assert_(False)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
