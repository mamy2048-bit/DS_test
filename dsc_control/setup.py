from glob import glob

from setuptools import find_packages, setup

package_name = "dsc_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="kmj",
    maintainer_email="isr3856@gmail.com",
    description="Velocity-tracking MPC + ESO inner loop for Nav2 on the M2 mobile robot",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mpc_node = dsc_control.nodes.mpc_node:main",
            "sim_robot = dsc_control.nodes.sim_robot:main",
        ],
    },
)
