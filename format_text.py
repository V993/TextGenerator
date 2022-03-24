from sys import argv

source_text = str(argv[1])

# data = open("./rep_speech.txt").read()
cleanedData = open("./cleaned_source_text/"+source_text[:-4]+"_cleaned.txt", 'w')

with open("./source_text/"+source_text,'r+') as file:
    # write only the nonempty lines
    for line in file:
        if not line.isspace():
            if '.' not in line:
                cleanedData.write(line)
                continue
            
            oneSentence = ""
            flag = 0
            for char in line:
                if flag: # skip the space after a .
                    flag = 0
                    continue
                if char == '.': # add the current sentence to the file, flag the following space.
                    cleanedData.write(oneSentence+'\n')
                    oneSentence = ""
                    flag = 1
                    continue

                oneSentence += char

            # cleanedData.write(line)
            

    #         cleanedData.write(line)

    # for each '.', replace with '\n', skipping following spaces
    # lastChar = ''
    # for character in file:
    #     if character == '.':
    #         cleanedData.write("\n")
    #     elif lastChar == '.':
    #         lastChar = character
    #         continue
    #     else:
    #         cleanedData.write(character)
    #     lastChar = character