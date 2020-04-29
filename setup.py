
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
        self.build_opencv()
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

    def build_opencv(self,):
        rootpth = os.getcwd()
        os.chdir('third_party/opencv')
        cwd = os.getcwd()
        os.makedirs('build')
        os.chdir('build')
        self.spawn(['cmake', '..', '-DCMAKE_BUILD_TYPE=RELEASE', '-DOPENCV_GENERATE_PKGCONFIG=ON', '-DWITH_TBB=ON', '-DBUILD_TBB=ON', '-GNinja'])
        self.spawn(['ninja', 'install'])
        os.chdir(cwd)
        shutil.rmtree('build')
        os.chdir(rootpth)


setup(
    name='cdataloader', # named used to install package egg/pip
    version='0.1',
    packages=['cdataloader'], # names to import in python
    ext_modules=[CMakeExtension('_cdataloader')], # whatever, no matter
    cmdclass={
        'build_ext': build_ext,
    },
    install_requires=['numpy >= 1.17.0'],
    python_requires='>=3.6',

)
