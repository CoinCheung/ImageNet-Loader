
import os
import os.path as osp
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_org


class CMakeExtension(Extension):

    def __init__(self, name):
        super().__init__(name, sources=[])


class build_ext(build_ext_org):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
            super().run()

    def build_cmake(self, ext):
        cwd = os.getcwd()
        build_dir = './build'
        if osp.exists(build_dir): shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        os.chdir(build_dir)
        self.spawn(['cmake', '..'])
        self.spawn(['make', '-j4'])
        os.chdir(cwd)
        shutil.rmtree(build_dir)


# TODO: install_requires, python_requires
setup(
    name='cdataloader', # named used to install package egg/pip
    version='0.1',
    packages=['cdataloader'], # names to import in python
    ext_modules=[CMakeExtension('_cdataloader')], # whatever, no matter
    cmdclass={
        'build_ext': build_ext,
    }
    install_requires=['numpy >= 1.17.0'],
    python_requires='>=3.6',

)
