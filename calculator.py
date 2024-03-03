while True:
    def add(x, y):
        return x + y

    def sub(x, y):
        return x - y

    def mult(x, y):
        return x * y

    def div(x, y):
        if y == 0:
            print("Cannot divide by 0")
            return None
        return x / y

    print("Calculator")
    print("For addition press 1")
    print("For subtraction press 2")
    print("For multiplication press 3")
    print("For division press 4")
    print("For exit press 5")

    choice = input("Choose an option (1/2/3/4/5): ")

    if choice in ('1', '2', '3', '4'):
        try:
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))
        except ValueError:
            print("Invalid input")
            continue

        if choice == '1':
            print(num1, "+", num2, "=", add(num1, num2))
        elif choice == '2':
            print(num1, "-", num2, "=", sub(num1, num2))
        elif choice == '3':
            print(num1, "x", num2, "=", mult(num1, num2))
        elif choice == '4':
            result = div(num1, num2)
            if result is not None:
                print(num1, "/", num2, "=", result)
    elif choice == '5':
        break
    else:
        print("Invalid choice. Please enter a valid option.")
