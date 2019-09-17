import tarfile
import os


def untar(input_dir, output_dir):
    for path, directories, files in os.walk(input_dir):
        for f in files:

            if f.endswith(".tar.gz"):
                index_of_dot = f.index('.')
                f_name_without_extension = f[:index_of_dot]
                tar = tarfile.open(os.path.join(path, f), 'r:gz')
                for member in tar.getmembers():
                    if member.name.endswith('.mtx'):  # skip if the TarInfo is not files
                        member.name = os.path.basename(member.name)  # remove the path by reset it
                        tar.extract(member, output_dir)  # extract
                tar.close()




untar('ss_small/', 'ss_small_set/')
