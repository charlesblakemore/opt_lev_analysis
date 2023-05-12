import sys, re, os

no_print = False  # Only the truly brave ever set this to True

dry_run = True
regex = True


file_ext = '.py'  # Make sure it's lowercase

# pattern = r'\{ch9:([^\n\}]*)\}'
# replace = r'{ch8:\1}'

# pattern = r'figures/chapter9'
# replace = r'figures/chapter8'

pattern = r'grav_util_3'
replace = r'grav_util'

# pattern = r'\\qty(\[?[^\n\]\}\{]*\]?)\{([^\n\}]*)\}\{([^\n\}]*)\^3([^\n\}]*)\}'
# replace = r'\\qty\1{\2}{\3\\cubed\4}'

# pattern = r'\\SI(\[?[^\n\}]*\]?)\{([^\n\}]*)\}\{([^\n\}]*)\}'
# replace = r'\\qty\1{\2}{\3}'

# pattern = r'\\SIproduct(\[?[^\n\}]*\]?)\{([^\n\}]*)\}\{([^\n\}]*)\}\{([^\n\}]*)\}'
# replace = r'\\qtyproduct\1{\2}{\3}{\4}'




### Needed for nice bolding in dry run. Couldn't figure out a good way to 
### programmatically generate this pattern from the replace string above

# replace_pattern = r'\{ch8:([^\n\}]*)\}'
# replace_pattern = r'figures/chapter8'
replace_pattern = r'grav_util'
# replace_pattern = r'\\qty(\[?[^\n\]\}\{]*\]?)\{([^\n\}]*)\}\{([^\n\}]*)\\cubed([^\n\}]*)\}'

# replace_pattern = r'\\qty(\[?[^\n\}]*\]?)\{([^\n\}]*)\}\{([^\n\}]*)\}'
# replace_pattern = r'\\qtyrange(\[?[^\n\}]*\]?)\{([^\n\}]*)\}\{([^\n\}]*)\}\{([^\n\}]*)\}'



###############################################################################
###############################################################################
###############################################################################



### Assumes this script is in the 'chapters' directory. This is some
### default safety shit so I don't find replace on other files anywhere
###
### WILL NEED TO BE CHANGED FOR OTHER PROJECTS AND/OR USERS
###    (the commented line may be helpful)
###
# directory = /path/to/my/favorite/place/with/files/
directory = os.path.dirname(os.path.abspath(__file__))
# assert os.path.basename(directory) == 'chapters', \
#         "Script isn't located in the 'chapters' subdirectory of the thesis"

### Find all the files with the requested extension in the given directory
files = [ f for f in os.listdir(directory) if \
            ( os.path.isfile(os.path.join(directory, f)) and (os.path.splitext(f)[-1].lower() == file_ext) ) ]
files.sort()



### Add these strings as prefixes to modify terminal printing behavior.
### They can be stacked for multiple modifications. Always pass the 
### color.END at the end of a modified string to return the terminal to 
### default font.
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'



def build_match_string(in_string, matches, modifier_string):
    '''Print the matches from a re.finditer() output (usually cast to a list),
       with the desired modifications for bolding and coloring etc.

       INPUTS:

            in_string - string to be printed with modifications

            matches - match objects from list(re.finditer()) enumerating
                desired locations for print modifiers

            modifier_string - desired modifiers packing into a single string,
                taken from the class attributes of the "color" class defined here
    '''

    out_string = in_string

    ### Proceeding backward from the last matching instance, add string
    ### printing modifiers to the matched text so it's easier to understand
    ### when printed to the terminal. Starting from the end and going back
    ### ensures the matching indices are still valid after each 
    ### successive modification.
    for match in matches[::-1]:
        ind1 = match.span()[0]
        ind2 = match.span()[1]
        out_string = out_string[:ind1] + modifier_string + out_string[ind1:ind2] \
                        + color.END + out_string[ind2:]

    return out_string



dry_run_string = color.BOLD + '#' * 67 + '\n' + '#########  ' \
                    + color.UNDERLINE + 'THIS WAS JUST A DRY-RUN. NO CHANGES WERE MADE' \
                    + color.END + color.BOLD + '  #########' + '\n' + '#' * 67 + color.END

if dry_run and not no_print:
    print()
    print(dry_run_string)
    print()

### Loop over each file found and perform the string replacement line-by-line
for file in files:

    if file in __file__:
        continue

    ### Large delimiter to separate files
    if not no_print:
        print('+' * 80)
        print('+++++++++  ' + color.BOLD + color.UNDERLINE \
                    + 'Looking in: < {:s} >'.format(file) + color.END)
        print()

    ### Read all the lines at once
    with open(file, 'r') as f:
        in_lines = f.readlines()

    ### Loop over the lines and check for the search pattern in each
    out_lines = []
    for line in in_lines:

        ### Quick check to see if it's there
        if regex:
            match = re.findall(pattern, line)
        else:
            match = (pattern in line)

        ### Append line and continue if no match was found and no 
        ### replacement is necessary
        if not match:
            out_lines.append(line)
            continue

        ### Perform the find/replace operation
        if regex:
            out = re.sub(pattern, replace, line)
        else:
            out = line.replace(pattern, replace)

        ### Append the new line to the output
        out_lines.append(out)

        if not no_print:
            ### Get the location of each of the matches in the original line
            matches = list(re.finditer(pattern, line))
            match_string = build_match_string(line, matches, color.BOLD+color.RED)
            print('     Original: ', match_string)

            ### Get the location of each of the matches in the modified line
            out_matches = list(re.finditer(replace_pattern, out))
            out_match_string = build_match_string(out, out_matches, color.BOLD+color.BLUE)
            print('          New: ', out_match_string)

            ### Delimiter between lines for slightly easier visual parsing
            print('     ' + '-' * 75)
            print()

    if not no_print:
        print()

    ### Write the output lines to the file if not a dry run
    if not dry_run:
        with open(file, 'w') as f:
            for line in out_lines:
                f.write(line)

if dry_run and not no_print:
    print()
    print(dry_run_string)
    print()
