fn main() {
    // Use the vendored protoc binary so no system protoc install is required.
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc not found");
    std::env::set_var("PROTOC", protoc);

    // Generate Rust gRPC stubs from the proto file at build time.
    tonic_build::compile_protos("proto/video.proto")
        .expect("Failed to compile proto files");

    tauri_build::build()
}
