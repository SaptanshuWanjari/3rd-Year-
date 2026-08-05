def xor(a, b):
    result = ""
    for i in range(1, len(b)):
        if a[i] == b[i]:
            result += "0"
        else:
            result += "1"
    return result


def mod2div(dividend, divisor):
    pick = len(divisor)
    tmp = dividend[:pick]

    while pick < len(dividend):
        if tmp[0] == "1":
            tmp = xor(divisor, tmp) + dividend[pick]
        else:
            tmp = xor("0" * pick, tmp) + dividend[pick]
        pick += 1

    if tmp[0] == "1":
        tmp = xor(divisor, tmp)
    else:
        tmp = xor("0" * pick, tmp)

    return tmp


data = input("Enter data frame: ")
divisor = input("Enter generator: ")

appended_data = data + "0" * (len(divisor) - 1)
remainder = mod2div(appended_data, divisor)
codeword = data + remainder

print("\nCRC:", remainder)
print("Transmitted:", codeword)
