import AVFoundation
import CoreML
import SwiftUI

/// Encapsulates audio capture and transcription. Alignment is optional; start with text-only.
final class TranscriberService: NSObject, ObservableObject {
    enum State {
        case idle
        case recording
        case transcribing
        case finished(String)
        case failed(Error)
    }

    @Published private(set) var state: State = .idle

    private let audioEngine = AVAudioEngine()
    private let session = AVAudioSession.sharedInstance()
    private let queue = DispatchQueue(label: "transcriber.queue", qos: .userInitiated)

    private var model: MLModel?
    private var buffer: AVAudioPCMBuffer?

    func configureSession() throws {
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
        try session.setPreferredSampleRate(16_000)
        try session.setPreferredInputNumberOfChannels(1)
        try session.setActive(true)
    }

    /// Load Core ML model from app bundle or sandbox. Swap to TorchModule if using PyTorch Lite.
    func loadModel(at url: URL) throws {
        model = try MLModel(contentsOf: url)
    }

    func startRecording() {
        do {
            try configureSession()
            let format = audioEngine.inputNode.inputFormat(forBus: 0)
            let targetRate: Double = 16_000

            audioEngine.inputNode.installTap(onBus: 0, bufferSize: 2048, format: format) { [weak self] buffer, _ in
                guard let self else { return }
                self.append(buffer: buffer, targetRate: targetRate)
            }

            audioEngine.prepare()
            try audioEngine.start()
            DispatchQueue.main.async { self.state = .recording }
        } catch {
            DispatchQueue.main.async { self.state = .failed(error) }
        }
    }

    func stopRecordingAndTranscribe() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)

        guard let buffer else {
            DispatchQueue.main.async { self.state = .failed(TranscriberError.noAudio) }
            return
        }

        DispatchQueue.main.async { self.state = .transcribing }
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let text = try self.runModel(buffer: buffer)
                DispatchQueue.main.async { self.state = .finished(text) }
            } catch {
                DispatchQueue.main.async { self.state = .failed(error) }
            }
        }
    }

    private func append(buffer incoming: AVAudioPCMBuffer, targetRate: Double) {
        // Resample to 16 kHz mono using AVAudioConverter + Accelerate when needed.
        let inputFormat = incoming.format
        guard inputFormat.sampleRate != targetRate || inputFormat.channelCount != 1 else {
            accumulate(incoming)
            return
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                                                                     sampleRate: targetRate,
                                                                                     channels: 1,
                                                                                     interleaved: false)!) else {
            return
        }

        let outputBuffer = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: incoming.frameCapacity)!
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            outStatus.pointee = .haveData
            return incoming
        }

        try? converter.convert(to: outputBuffer, error: nil, withInputFrom: inputBlock)
        accumulate(outputBuffer)
    }

    private func accumulate(_ pcmBuffer: AVAudioPCMBuffer) {
        if buffer == nil {
            buffer = pcmBuffer
        } else {
            buffer?.append(pcmBuffer)
        }
    }

    private func runModel(buffer: AVAudioPCMBuffer) throws -> String {
        guard let model else { throw TranscriberError.modelMissing }
        let frameLength = Int(buffer.frameLength)
        guard let channelData = buffer.floatChannelData?.pointee else { throw TranscriberError.invalidBuffer }

        // Normalize samples to [-1, 1]
        let audioArray = Array(UnsafeBufferPointer(start: channelData, count: frameLength))
        let mlArray = try MLMultiArray(shape: [NSNumber(value: audioArray.count)], dataType: .float32)
        for (idx, sample) in audioArray.enumerated() {
            mlArray[idx] = NSNumber(value: sample)
        }

        let input = MLDictionaryFeatureProvider(dictionary: ["audio_features": mlArray])
        let prediction = try model.prediction(from: input)
        if let text = prediction.featureValue(for: "transcription")?.stringValue {
            return text
        }

        return ""
    }
}

enum TranscriberError: Error {
    case modelMissing
    case invalidBuffer
    case noAudio
}
