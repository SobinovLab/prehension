#!python3
# -*- coding: utf-8 -*-
"""
Working with filenames and copying of files.

Copyright (C) 2019-2024 Anton Sobinov
https://github.com/BensmaiaLab/prehension

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import shutil
import datetime
import time
import re
from glob import glob

from .logs import rs, ws


# @https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb
def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)
    else:
        return '{0:.2f} TB'.format(B / TB)


def print_copy(*args, **kwargs):
    """Simulates shutil.copy, but only prints out the targets."""
    print('{} -> {}'.format(args[0], args[1]))


class PrintCopyAccumulateSize():
    """Calculates the sum of sizes of files passed as  """

    def __init__(self, dry_run, verbose):
        self.size = 0
        self.dry_run = dry_run
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        if self.verbose > 0:
            print_copy(*args, **kwargs)
        self.size += os.path.getsize(args[0])
        if not self.dry_run:
            shutil.copy2(*args, **kwargs)

    def __str__(self):
        return humanbytes(self.size)


def copytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copy2,
             ignore_dangling_symlinks=False, make_dir=True, dir_exist_ok=False,
             overwrite_existing=None, verbose=False):
    """Recursively copy a directory tree.

    ADAPTED from shutil.py function. Changes:
        option to turn off make directory. Useful when only walking the tree with reports instead
            of copy_functions. If actually trying to copy the files, will most likely throw an
            error.
        option to make directory exist_ok true
        a callable option that asks what to do when the file already exists at the destination.
            Return True to overwrite, False to skip.
        verbose prints current source directory.

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied. If the file pointed by the symlink doesn't
    exist, an exception will be added in the list of errors raised in
    an Error exception at the end of the copy process.

    You can set the optional ignore_dangling_symlinks flag to true if you
    want to silence this exception. Notice that this has no effect on
    platforms that don't support os.symlink.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    The optional copy_function argument is a callable that will be used
    to copy each file. It will be called with the source path and the
    destination path as arguments. By default, copy2() is used, but any
    function that supports the same signature (like copy()) can be used.

    """
    if verbose:
        print(src)
    names = sorted(os.listdir(src))
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()

    if make_dir:
        os.makedirs(dst, exist_ok=dir_exist_ok)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.islink(srcname):
                linkto = os.readlink(srcname)
                if symlinks:
                    # We can't just leave it to `copy_function` because legacy
                    # code with a custom `copy_function` may rely on copytree
                    # doing the right thing.
                    os.symlink(linkto, dstname)
                    shutil.copystat(srcname, dstname,
                                    follow_symlinks=not symlinks)
                else:
                    # ignore dangling symlink if the flag is on
                    if not os.path.exists(linkto) and ignore_dangling_symlinks:
                        continue
                    # otherwise let the copy occurs. copy2 will raise an error
                    if os.path.isdir(srcname):
                        copytree(
                            srcname, dstname,
                            symlinks=symlinks, ignore=ignore, copy_function=copy_function,
                            ignore_dangling_symlinks=ignore_dangling_symlinks, make_dir=make_dir,
                            dir_exist_ok=dir_exist_ok, overwrite_existing=overwrite_existing,
                            verbose=verbose)
                    else:
                        if (overwrite_existing is None or not os.path.exists(dstname) or
                                overwrite_existing(srcname, dstname)):
                            copy_function(srcname, dstname)
            elif os.path.isdir(srcname):
                copytree(
                    srcname, dstname,
                    symlinks=symlinks, ignore=ignore, copy_function=copy_function,
                    ignore_dangling_symlinks=ignore_dangling_symlinks, make_dir=make_dir,
                    dir_exist_ok=dir_exist_ok, overwrite_existing=overwrite_existing,
                    verbose=verbose)
            else:
                if (overwrite_existing is None or not os.path.exists(dstname) or
                        overwrite_existing(srcname, dstname)):
                    # Will raise a SpecialFileError for unsupported file types
                    copy_function(srcname, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except shutil.Error as err:
            errors.extend(err.args[0])
        except OSError as why:
            errors.append((srcname, dstname, str(why)))
    try:
        shutil.copystat(src, dst)
    except OSError as why:
        # Copying file access times may fail on Windows
        if getattr(why, 'winerror', None) is None:
            errors.append((src, dst, str(why)))
    if errors:
        raise shutil.Error(errors)
    return dst


def overwrite_existing_always(srcname, dstname):
    return True


def overwrite_existing_never(srcname, dstname):
    return False


def overwrite_existing_new(srcname, dstname):
    return os.path.getmtime(srcname) > os.path.getmtime(dstname)


def overwrite_existing_new_box(srcname, dstname):
    # BOX does not save sub-second creation time
    return os.path.getmtime(srcname) - os.path.getmtime(dstname) >= 1


def copy_folder_contents(src_dir, target_dir, dir_names=[], file_names=[], suppress_warnings=False,
                         dry_run=False, copy_function=None, box=False, overwrite=False):
    start_time = time.time()
    if copy_function is None:
        local_copy_function = True
        copy_function = PrintCopyAccumulateSize(dry_run, int(dry_run))
    else:
        local_copy_function = False

    if overwrite:
        oex = overwrite_existing_always
    elif box:
        oex = overwrite_existing_new_box
    else:
        oex = overwrite_existing_new

    for dir_name in dir_names:
        src = os.path.join(src_dir, dir_name)
        dst = os.path.join(target_dir, dir_name)
        if os.path.isdir(src):
            copytree(src, dst, overwrite_existing=oex, dir_exist_ok=True,
                     copy_function=copy_function)
        else:
            if not suppress_warnings:
                ws(f'Warning: could not find expected directory ({src}) to upload')

    for fname in file_names:
        f_src = os.path.join(src_dir, fname)
        f_dst = os.path.join(target_dir, fname)

        if not os.path.isfile(f_src):
            if not suppress_warnings:
                ws(f'Warning: could not find expected file ({f_src}) to upload')
            continue

        # Check if the destination file exists
        if not os.path.isfile(f_dst) or oex(f_src, f_dst):
            # Destination file does not exist, copy the source file
            copy_function(f_src, f_dst)

    timedelta = str(datetime.timedelta(
        seconds=time.time() - start_time))
    # if shutil copy, the report won't be meaningful
    if local_copy_function:
        rs('Found {} files for copying during {}.'.format(
            copy_function, timedelta))


def get_image_list(path=None, sort=True):
    '''Returns a list of all image filenames.

    Wrapper for 'utilities.get_file_list' with file_extensions defined as:
        ('jpg', 'jpeg', 'png', 'bmp')

    Keyword Arguments:
        path {string} -- directory to explore. (default: current directory)
        sort {bool} -- alphanumeric sort the output list (default: {True})
    Output:
        strings {list of strings} -- list of all image filenames.
    '''

    return get_file_list(('jpg', 'jpeg', 'png', 'bmp'), path=path, sort=sort)


def get_file_list(file_extensions, path=None, sort=True):
    '''Returns a list of all filenames with a specific extension.

    Files with extensions .jpg, .jpeg, .png, .bmp are considered images. Searches shell-style for
        <path>/*.<file extension>

    Arguments:
        file_extensions {list of strings} -- file extensions to return, with or without the dot.
            If None or empty, returns all files with extensions.
    Keyword Arguments:
        path {string} -- directory to explore. (default: current directory)
        sort {bool} -- alphanumeric sort the output list (default: {True})
    Output:
        strings {list of strings} -- list of all filenames with specified extension.
    '''
    if isinstance(file_extensions, str):
        file_extensions = [file_extensions]

    if file_extensions is None or len(file_extensions) == 0:
        file_extensions = ('*', )
    else:
        file_extensions = [ifx.strip('.') for ifx in file_extensions]

    files = []
    for file_extension in file_extensions:
        if path is None:
            files += glob('*.' + file_extension)
        else:
            files += glob(os.path.join(path, '*.' + file_extension))

    if sort:
        files = alphanumeric_sort(files)

    return files


def alphanumeric_sort(strings):
    '''Alphanumeric sorter that considers parts of the numerical parts of the string independently.

    For example, 'text9moretext' < 'text10moretext' when using this sorting function.
    Useful for:
        sorting out very high framerate images
        sorting by framerate, because '11'<'9', but 11>9
        recording for more than 999.9999 seconds (in the current format the generic sort does not
            give the desired result).

    Arguments:
        strings {list of strings} -- list of strings (e.g. filenames to be sorted)
    Output:
        strings {list of strings} -- sorted list of strings
    '''
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(strings, key=alphanum_key)
