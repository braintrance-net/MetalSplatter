import XCTest
import Metal
import simd
@testable import MetalSplatter

final class SplatAnimationTests: XCTestCase {
    typealias Uniforms = SplatRenderer.Uniforms

    private func makeUniforms() -> Uniforms {
        Uniforms(projectionMatrix: matrix_identity_float4x4,
                 viewMatrix: matrix_identity_float4x4,
                 cameraPosition: MTLPackedFloat3Make(1, 2, 3),
                 screenSize: SIMD2(800, 600),
                 focalX: 400,
                 focalY: 300,
                 tanHalfFovX: 1,
                 tanHalfFovY: 1,
                 chunkCount: 1,
                 splatCount: 100,
                 indexedSplatCount: 64)
    }

    private func bytes(of uniforms: Uniforms) -> [UInt8] {
        withUnsafeBytes(of: uniforms) { Array($0) }
    }

    /// Member offsets and array stride of `Uniforms` as the compiled Metal library sees them.
    private func metalUniformsLayout() throws -> (offsets: [String: Int], stride: Int) {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let library = try device.makeDefaultLibrary(bundle: Bundle.module)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = try XCTUnwrap(library.makeFunction(name: "singleStageSplatVertexShader"))
        descriptor.fragmentFunction = try XCTUnwrap(library.makeFunction(name: "singleStageSplatFragmentShader"))
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        let (_, reflection) = try device.makeRenderPipelineState(descriptor: descriptor,
                                                                 options: [.bindingInfo, .bufferTypeInfo])
        let binding = try XCTUnwrap(reflection?.vertexBindings
            .compactMap { $0 as? MTLBufferBinding }
            .first { $0.index == SplatRenderer.BufferIndex.uniforms.rawValue })
        let array = try XCTUnwrap(binding.bufferStructType?.memberByName("uniforms")?.arrayType())
        let element = try XCTUnwrap(array.elementStructType())
        let offsets = Dictionary(uniqueKeysWithValues: element.members.map { ($0.name, $0.offset) })
        return (offsets, array.stride)
    }

    func testSwiftUniformsLayoutMatchesMetal() throws {
        let metal = try metalUniformsLayout()

        XCTAssertEqual(MemoryLayout<Uniforms>.stride, metal.stride)
        XCTAssertEqual(MemoryLayout<SplatRenderer.UniformsArray>.offset(of: \.uniforms1), metal.stride)

        let swiftOffsets: [String: Int?] = [
            "projectionMatrix": MemoryLayout<Uniforms>.offset(of: \.projectionMatrix),
            "viewMatrix": MemoryLayout<Uniforms>.offset(of: \.viewMatrix),
            "cameraPosition": MemoryLayout<Uniforms>.offset(of: \.cameraPosition),
            "screenSize": MemoryLayout<Uniforms>.offset(of: \.screenSize),
            "focalX": MemoryLayout<Uniforms>.offset(of: \.focalX),
            "tanHalfFovY": MemoryLayout<Uniforms>.offset(of: \.tanHalfFovY),
            "chunkCount": MemoryLayout<Uniforms>.offset(of: \.chunkCount),
            "splatCount": MemoryLayout<Uniforms>.offset(of: \.splatCount),
            "indexedSplatCount": MemoryLayout<Uniforms>.offset(of: \.indexedSplatCount),
            "animationCenterAndRadius": MemoryLayout<Uniforms>.offset(of: \.animationCenterAndRadius),
            "animationParams": MemoryLayout<Uniforms>.offset(of: \.animationParams),
            "animationMode": MemoryLayout<Uniforms>.offset(of: \.animationMode),
            "_animationPadding": MemoryLayout<Uniforms>.offset(of: \._animationPadding),
        ]
        for (name, offset) in swiftOffsets {
            XCTAssertEqual(offset, metal.offsets[name], "offset of \(name)")
        }
    }

    func testAnimationEncodesIntoUniformBytes() throws {
        var uniforms = makeUniforms()
        uniforms.setAnimation(SplatRenderer.SplatAnimation(center: SIMD3(1, 2, 3),
                                                           radius: 4,
                                                           progress: 0.5,
                                                           feather: 0.1,
                                                           pointScale: 0.25,
                                                           pointAlpha: 0.6))
        let raw = bytes(of: uniforms)
        func float(at offset: Int) -> Float {
            raw.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: Float.self) }
        }

        let metal = try metalUniformsLayout()
        let centerOffset = try XCTUnwrap(metal.offsets["animationCenterAndRadius"])
        let paramsOffset = try XCTUnwrap(metal.offsets["animationParams"])
        let modeOffset = try XCTUnwrap(metal.offsets["animationMode"])

        XCTAssertEqual((0..<4).map { float(at: centerOffset + $0 * 4) }, [1, 2, 3, 4])
        XCTAssertEqual((0..<4).map { float(at: paramsOffset + $0 * 4) }, [0.5, 0.1, 0.25, 0.6])
        XCTAssertEqual(raw.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: modeOffset, as: UInt32.self) }, 1)
    }

    func testInitClampsToShaderRanges() {
        let animation = SplatRenderer.SplatAnimation(center: .zero,
                                                     radius: 0,
                                                     progress: 2,
                                                     feather: -1,
                                                     pointScale: 5,
                                                     pointAlpha: -1)
        XCTAssertEqual(animation.radius, 0.0001)
        XCTAssertEqual(animation.progress, 1)
        XCTAssertEqual(animation.feather, 0.0001)
        XCTAssertEqual(animation.pointScale, 1)
        XCTAssertEqual(animation.pointAlpha, 0)
    }

    func testNilAnimationLeavesExistingFieldsAndZeroesTail() throws {
        let baseline = makeUniforms()
        var cleared = makeUniforms()
        cleared.setAnimation(SplatRenderer.SplatAnimation(center: SIMD3(9, 9, 9), radius: 9, progress: 0.3))
        cleared.setAnimation(nil)

        // Fields that existed before the animation tail keep their pre-change offsets.
        XCTAssertEqual(MemoryLayout<Uniforms>.offset(of: \.indexedSplatCount), 176)
        let tailOffset = try XCTUnwrap(MemoryLayout<Uniforms>.offset(of: \.animationCenterAndRadius))
        XCTAssertEqual(tailOffset, 192)

        // Compiler padding between the two ranges is not guaranteed to be zeroed, so it is skipped.
        let existingFields = 0..<180
        XCTAssertEqual(bytes(of: cleared)[existingFields], bytes(of: baseline)[existingFields])
        XCTAssertTrue(bytes(of: cleared)[tailOffset..<MemoryLayout<Uniforms>.size].allSatisfy { $0 == 0 })
        XCTAssertEqual(cleared.animationMode, 0)
    }
}
