import XCTest
import Metal
import simd
@testable import MetalSplatter

final class LocalitySHTests: XCTestCase {
    func testLocalityKeepsEachSplatWithItsSHCoefficients() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let count = 257
        let points = try MetalBuffer<EncodedSplatPoint>(device: device, capacity: count)
        let sh = try MetalBuffer<Float16>(device: device, capacity: count * 45)
        for index in 0..<count {
            let f = Float(index)
            points.append(EncodedSplatPoint(position: SIMD3(sin(f * 7), cos(f * 3), sin(f * 11)),
                          colorSH0: SIMD3(f, 0, 0), opacity: 1,
                          scale: SIMD3(repeating: 0.1), rotation: simd_quatf()))
            for coefficient in 0..<45 { sh.append(Float16(index * 100 + coefficient)) }
        }
        let chunk = SplatChunk(splats: points, shCoefficients: sh, shDegree: .sh3)
        chunk.sortByLocality()
        XCTAssertTrue((0..<count).contains { Int(points.values[$0].colorSH0.r) != $0 },
                      "The fixture must exercise an actual locality permutation")
        for destination in 0..<count {
            let source = Int(points.values[destination].colorSH0.r)
            for coefficient in 0..<45 {
                XCTAssertEqual(sh.values[destination * 45 + coefficient], Float16(source * 100 + coefficient))
            }
        }
    }
}
