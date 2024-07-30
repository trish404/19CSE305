def calcrange(nums):
    if len(nums) < 3:
        return "Range determination not possible"
    else:
        minv = min(nums)
        maxv = max(nums)
        return maxv - minv

inp = input("Enter a list of real numbers separated by spaces: ")
nums = list(map(float, inp.split()))
res = calcrange(nums)
print(res)
