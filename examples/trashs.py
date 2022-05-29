
import dragonfly

import multiprocessing


def func(a):
    return a + a


class Foo:
    def __init__(self, func_):
        self.func_ = func_

    def evaluate(self, a):
        return self.func_(a)

    def evaluate_multi(self, a):
        with multiprocessing.Pool(2) as pool:
            results = pool.map(foo.evaluate, a)
        return results


class Foo2:
    def __init__(self, sub_class: Foo):
        self.sub_class = sub_class
        self.issue = self.do_stuff()

    def do_stuff(self):
        domain = [{'name': 'temperature', 'type': 'float',  'min': 40, 'max': 100}]
        from dragonfly import load_config
        return load_config({'domain': domain})

    def evaluate(self):
        a = [1,2,3,4,5,6,7,8,9]
        print(self.sub_class.evaluate_multi(a))


if __name__ == '__main__':
    foo = Foo(func)
    foo2 = Foo2(foo)

    foo2.evaluate()

