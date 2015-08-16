cdef extern from "<random>" namespace "std" nogil:
    # workaround: https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
    cdef cppclass mersenne_twister_engine \
        "std::mersenne_twister_engine<class UIntType, size_t w, size_t n, size_t m, size_t r, UIntType a, size_t u, UIntType d, size_t s, UIntType b, size_t t, UIntType c, size_t l, UIntType f>"[UIntType]:
        mersenne_twister_engine(UIntType)
    ctypedef mersenne_twister_engine[int] mt19937 "std::mersenne_twister_engine<uint_fast32_t, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7, 0x9d2c5680, 15, 0xefc60000, 18, 1812433253>"

    cdef cppclass random_device:
        random_device()
        unsigned int operator()()

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution(T, T) except +
        T operator()[U](U&);


