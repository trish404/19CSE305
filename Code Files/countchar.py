def hoc(input_str):
    freq = {}
    
    for char in input_str:
        if char.isalpha():
            if char in freq:
                freq[char] += 1
            else:
                freq[char] = 1
    
    maxch = max(freq, key=freq.get)
    macc = freq[maxch]
    
    return maxch, macc

inp = input("Enter a string: ")

char, count = hoc(inp)

print(f"The most occurring character is '{char}' & occurrs {count} times.")
