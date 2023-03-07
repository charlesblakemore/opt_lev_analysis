import os, sys, re, fnmatch, traceback

from bead_data_funcs import get_hdf5_time




def find_all_fnames(dirlist, ext='.h5', sort=False, exclude_fpga=True, \
                    verbose=True, substr='', subdir='', sort_time=False, \
                    sort_by_index=False, use_origin_timestamp=False, \
                    skip_subdirectories=False):
    '''Finds all the filenames matching a particular extension
       type in the directory and its subdirectories .

       INPUTS: 
            dirlist, list of directory names to loop over

            ext, file extension you're looking for
            
            sort, boolean specifying whether to do a simple sort
            
            exclude_fpga, boolean to ignore hdf5 files directly from
                the fpga given that the classes load these automatically
               
            verbose, print shit
            
            substr, string within the the child filename to match
            
            subdir, string within the full parent director to match
               
            sort_time, sort files by timestamp, trying to use the hdf5 
                timestamp first

            use_origin_timestamp, boolean to tell the time sorting to
                use the file creation timestamp itself

            skip_subdirectories, boolean to skip recursion into child
                directories when data is organized poorly

       OUTPUTS: 
            files, list of filenames as strings, or list of lists if 
                multiple input directories were given

            lengths, length of file lists found '''

    if verbose:
        print("Finding files in: ")
        print(dirlist)
        sys.stdout.flush()

    was_list = True

    lengths = []
    files = []

    if type(dirlist) == str:
        dirlist = [dirlist]
        was_list = False

    for dirname in dirlist:
        for root, dirnames, filenames in os.walk(dirname):
            if (root != dirname) and skip_subdirectories:
                continue
            for filename in fnmatch.filter(filenames, '*' + ext):
                if ('_fpga.h5' in filename) and exclude_fpga:
                    continue
                if substr and (substr not in filename):
                    continue
                if subdir and (subdir not in root):
                    continue
                files.append(os.path.join(root, filename))
        if was_list:
            if len(lengths) == 0:
                lengths.append(len(files))
            else:
                lengths.append(len(files) - np.sum(lengths)) 

    if len(files) == 0:
        print("DIDN'T FIND ANY FILES :(")

    if 'new_trap' in files[0]:
        new_trap = True
    else:
        new_trap = False

    if sort:
        # Sort files based on final index
        files.sort(key = find_str)

    if sort_by_index:
        files.sort(key = lambda x: int(re.findall(r'_([0-9]+)\.', x)[0]) )

    if sort_time:
        files = sort_files_by_timestamp(files, use_origin_timestamp=use_origin_timestamp, \
                                        new_trap=new_trap)

    if verbose:
        print("Found %i files..." % len(files))
    if was_list:
        return files, lengths
    else:
        return files, [len(files)]



def sort_files_by_timestamp(files, use_origin_timestamp=False, new_trap=False):
    '''Pretty self-explanatory function.'''

    if not use_origin_timestamp:
        try:
            files = [(get_hdf5_time(path, new_trap=new_trap), path) for path in files]
        except Exception:
            print('BAD HDF5 TIMESTAMPS, USING GENESIS TIMESTAMP')
            traceback.print_exc()
            use_origin_timestamp = True

    if use_origin_timestamp:
        files = [(os.stat(path), path) for path in files]
        files = [(stat.st_ctime, path) for stat, path in files]

    files.sort(key = lambda x: x[0])
    files = [obj[1] for obj in files]
    return files



def unpack_config_dict(dic, vec):
    '''takes vector containing data atributes and puts 
       it into a dictionary with key value pairs specified 
       by dict where the keys of dict give the labels and 
       the values specify the index in vec'''
    out_dict = {}
    for k in list(dic.keys()):
        out_dict[k] = vec[dic[k]]
    return out_dict 


