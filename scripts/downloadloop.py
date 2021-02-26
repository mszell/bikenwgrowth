debug = False

exec(open("../parameters/parameters.py").read())
exec(open("../code/path.py").read())
exec(open("../code/setup.py").read())
exec(open("../code/functions.py").read())

if __name__ == '__main__':

    print("Running 01.py")
    exec(open("01.py").read())

    print("Running 02.py")
    exec(open("02.py").read())