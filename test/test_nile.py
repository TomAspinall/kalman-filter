from test._base import KalmanFilterBaseTest

test = KalmanFilterBaseTest("test/data/nile")

if __name__ == "__main__":
    test.run_test()
    print("Nile test success!")
