def countpairs(lst, sumoflst):
    f = {}
    count = 0
    for num in lst:
        comp = sumoflst - num
        if comp in f:
            count += f[comp]
        if num in f:
            f[num] += 1
        else:
            f[num] = 1
    return count

lst = [2, 7, 4, 1, 3, 6]
sumoflst = 10

noofpairs = countpairs(lst, sumoflst)
print(f"No. of pairs w/ sum {sumoflst}: {noofpairs}")