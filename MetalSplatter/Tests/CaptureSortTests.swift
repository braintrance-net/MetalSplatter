import XCTest
import Metal
import simd
@testable import MetalSplatter

final class CaptureSortTests: XCTestCase {
    func testCaptureRejectsSequentialFallback() async throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let sorter = try SplatSorter(device: device)
        let gate = DispatchSemaphore(value: 0)
        defer { gate.signal() }
        let started = expectation(description: "sort starts")
        let finished = expectation(description: "sort finishes")
        sorter.onSortStart = { started.fulfill(); gate.wait() }
        sorter.onSortComplete = { _ in finished.fulfill() }
        let points = try MetalBuffer<EncodedSplatPoint>(device: device, capacity: 2)
        for z: Float in [-1, -5] {
            points.append(EncodedSplatPoint(position: SIMD3(0, 0, z), colorSH0: .zero,
                         opacity: 1, scale: SIMD3(repeating: 0.1), rotation: simd_quatf()))
        }
        try sorter.addChunkToSort(.init(chunkIndex: 0, buffer: points))
        await fulfillment(of: [started], timeout: 5)
        let fallback = try XCTUnwrap(sorter.tryObtainSortedIndices())
        XCTAssertEqual(fallback.values[0].splatIndex, 0)
        sorter.releaseSortedIndices(fallback)
        XCTAssertNil(sorter.tryObtainSortedIndices(requireCurrentSort: true))
        gate.signal()
        await fulfillment(of: [finished], timeout: 5)
        let sorted = try XCTUnwrap(sorter.tryObtainSortedIndices(requireCurrentSort: true))
        XCTAssertEqual(sorted.values[0].splatIndex, 1)
        sorter.releaseSortedIndices(sorted)

        // An old-camera result must be rejected even though it remains valid for live UI.
        let cameraStarted = expectation(description: "camera sort starts")
        let cameraFinished = expectation(description: "camera sort finishes")
        sorter.onSortStart = { cameraStarted.fulfill(); gate.wait() }
        sorter.onSortComplete = { _ in cameraFinished.fulfill() }
        sorter.updateCameraPose(position: SIMD3(0, 0, -10), forward: SIMD3(0, 0, 1))
        await fulfillment(of: [cameraStarted], timeout: 5)
        XCTAssertNil(sorter.tryObtainSortedIndices(requireCurrentSort: true))
        gate.signal()
        await fulfillment(of: [cameraFinished], timeout: 5)
        let moved = try XCTUnwrap(sorter.tryObtainSortedIndices(requireCurrentSort: true))
        XCTAssertEqual(moved.values[0].splatIndex, 0)
        sorter.releaseSortedIndices(moved)

        // Appending a chunk patches the published buffer with unsorted entries.
        let appendStarted = expectation(description: "append sort starts")
        let appendFinished = expectation(description: "append sort finishes")
        sorter.onSortStart = { appendStarted.fulfill(); gate.wait() }
        sorter.onSortComplete = { _ in appendFinished.fulfill() }
        try sorter.addChunkToSort(.init(chunkIndex: 1, buffer: points))
        await fulfillment(of: [appendStarted], timeout: 5)
        XCTAssertNil(sorter.tryObtainSortedIndices(requireCurrentSort: true))
        gate.signal()
        await fulfillment(of: [appendFinished], timeout: 5)
        let appended = try XCTUnwrap(sorter.tryObtainSortedIndices(requireCurrentSort: true))
        XCTAssertEqual(appended.count, 4)
        sorter.releaseSortedIndices(appended)
        sorter.setChunks([])
    }
}
