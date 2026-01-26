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
        'console_scripts': ['main_jetson = rescue_system.main_jetson:main',
                            'webcam_pub = rescue_system.webcam_publisher:main',
                            'pos_est_test = rescue_system.pos_est_test:main',
                            'tf_broadcaster = rescue_system.tf_broadcaster:main ',
                            'camera = rescue_system.camera_node:main'
        ],

    },
)
