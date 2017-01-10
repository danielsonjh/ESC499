import os
import shutil
import logging

input_dir = '../ModelNet10'
size = 90
output_dir = input_dir + '_binvox_' + str(size)
file_ext = '.off'


def main():
    logging.basicConfig(filename=output_dir + 'corrupt_files.log', level=logging.WARNING)

    for root, dirs, files in os.walk(input_dir):
        output_root = root.replace(input_dir, output_dir, 1)
        for in_file in files:
            if in_file.endswith(file_ext):
                out_file = replace_ext(in_file, file_ext, '.binvox')

                input_file_path = os.path.join(root, in_file)
                temp_file_path = os.path.join(root, out_file)
                output_file_path = os.path.join(output_root, out_file)

                if not os.path.exists(output_root):
                    os.makedirs(output_root)

                try:
                    os.system('binvox -e -cb -d ' + str(size) + ' ' + input_file_path)
                    shutil.move(temp_file_path, output_file_path)
                except IOError as e:
                    logging.warning('Error converting: ' + input_file_path)


def replace_ext(file_name, ext1, ext2):
    return file_name[::-1].replace(ext1[::-1], ext2[::-1], 1)[::-1]


main()