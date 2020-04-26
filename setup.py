
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

setup(
    name='dataloader',
    version='0.1',
    packages=['coin'],
    ext_modules=[CMakeExtension()],
    cmdclass={
        'build_ext': build_ext,
    }
)
