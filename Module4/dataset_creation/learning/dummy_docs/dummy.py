# This file was created to test the file reader.py and understand how it works

print("Hello World")

def compute_fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return compute_fibonacci(n - 1) + compute_fibonacci(n - 2)
n = 5
print(f"This is the {n} term in the fibonacci sequence: {compute_fibonacci(n)}")