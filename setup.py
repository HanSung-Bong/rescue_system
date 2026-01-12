from setuptools import find_packages, setup

package_name = 'rescue_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hansung',
    maintainer_email='peterpen0110@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ['main_pc = rescue_system.main_pc:main',
                            'webcam_pub = rescue_system.webcam_publisher:main',
        ],

    },
)
