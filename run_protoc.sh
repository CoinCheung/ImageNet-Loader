
src_dir=./
proto=interface.proto
out_dir=./comm
py_out_dir=./commpy

mkdir -p $out_dir $py_out_dir

protoc -I=$src_dir --grpc_out=$out_dir --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` $src_dir/$proto
protoc -I=$src_dir --cpp_out=$out_dir $src_dir/$proto


python -m grpc_tools.protoc -I=$src_dir --python_out=$py_out_dir --grpc_python_out=$py_out_dir $src_dir/$proto
