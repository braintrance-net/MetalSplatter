import XCTest
import Metal
import simd

final class ProjectionClipTests: XCTestCase {
    func testPlayCanvasKeepsFrontFacingSplatsAcrossClipPlanes() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let resources = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Resources")
        // Compile the production helper, rather than duplicating projection math.
        var source = "#include <metal_stdlib>\n#include <simd/simd.h>\n"
        for file in ["ShaderCommon.h", "SplatProcessing.h", "SplatProcessing.metal"] {
            let text = try String(contentsOf: resources.appendingPathComponent(file), encoding: .utf8)
            source += text.split(separator: "\n", omittingEmptySubsequences: false)
                .filter { !$0.hasPrefix("#import") && !$0.hasPrefix("#include") }.joined(separator: "\n") + "\n"
        }
        source += """
        kernel void testClipPlanes(device float4* results [[buffer(0)]]) {
            Uniforms u = {};
            u.viewMatrix = float4x4(1.0f);
            float nearZ = 0.01f, farZ = 10.0f;
            u.projectionMatrix = float4x4(float4(1,0,0,0), float4(0,1,0,0),
                float4(0,0,farZ/(nearZ-farZ),-1), float4(0,0,nearZ*farZ/(nearZ-farZ),0));
            u.screenSize = uint2(128);
            u.focalX = u.focalY = 64;
            u.tanHalfFovX = u.tanHalfFovY = 1;
            Splat s = {};
            s.color = half4(0,0,0,1);
            s.covA = packed_half3(0.000001h,0,0);
            s.covB = packed_half3(0.000001h,0,0.000001h);
            float depths[] = {-0.005f, -20.0f, 0.005f, -0.0001f};
            for (uint i=0; i<4; ++i) {
                s.position = packed_float3(0,0,depths[i]);
                u.renderingFlags = 2;
                results[i] = splatVertex(s,u,0,nullptr,SHDegree0,0).position;
                u.renderingFlags = 0;
                results[i+4] = splatVertex(s,u,0,nullptr,SHDegree0,0).position;
            }
        }
        """
        let library = try device.makeLibrary(source: source, options: nil)
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "testClipPlanes")))
        let output = try XCTUnwrap(device.makeBuffer(length: 8 * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, String(describing: command.error))
        let vertices = output.contents().bindMemory(to: SIMD4<Float>.self, capacity: 8)
        XCTAssertEqual(vertices[0].w, 0.005, accuracy: 1e-7)
        XCTAssertEqual(vertices[0].z, 0)
        XCTAssertEqual(vertices[1].w, 20)
        XCTAssertEqual(vertices[1].z, 20)
        XCTAssertEqual(vertices[2], SIMD4(1,1,0,1)) // Behind the camera stays culled.
        XCTAssertEqual(vertices[3].w, 0.0001, accuracy: 1e-8)
        XCTAssertTrue((0..<4).allSatisfy { vertices[$0].x.isFinite && vertices[$0].y.isFinite })
        XCTAssertEqual(vertices[4], SIMD4(1,1,0,1)) // Standard kernel retains near/far clipping.
        XCTAssertEqual(vertices[5], SIMD4(1,1,0,1))
    }
}
