#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include<iostream>

namespace py = pybind11;

class MyClass {
    Eigen::MatrixXd big_mat; // = Eigen::MatrixXd::Zero(10000, 10000);
    Eigen::MatrixXd other_mat;
public:
    MyClass(Eigen::MatrixXd &M, Eigen::MatrixXd &N):big_mat(M), other_mat(N)
    {
        std::cout << "M\n" << &M << std::endl;
        std::cout << "big_mat\n" << &big_mat << std::endl;
        
    };
    Eigen::MatrixXd &getMatrix() {
        std::cout << "big_mat2\n" << &big_mat << std::endl;     
         return big_mat; 
    }
    const Eigen::MatrixXd &viewMatrix() { return big_mat; }
    void mult(const Eigen::Ref<const Eigen::MatrixXd> v)
    {
        v = v * 2;
    }
};
        
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 eigen example plugin"; // optional module docstring

    py::class_<MyClass>(m, "MyClass")
        .def(py::init<Eigen::MatrixXd, Eigen::MatrixXd>())
        .def("copy_matrix", &MyClass::getMatrix) // Makes a copy!
        .def("get_matrix", &MyClass::getMatrix, py::return_value_policy::reference_internal)
        .def("view_matrix", &MyClass::viewMatrix, py::return_value_policy::reference_internal)
        .def("mult", &MyClass::mult, py::return_value_policy::reference_internal)
        ;
}