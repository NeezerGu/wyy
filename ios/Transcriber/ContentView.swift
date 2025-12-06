import SwiftUI

struct ContentView: View {
    @StateObject private var service = TranscriberService()
    @State private var status: String = "Ready"

    var body: some View {
        VStack(spacing: 16) {
            Text("WhisperX On-Device Demo")
                .font(.title2)

            Text(status)
                .foregroundStyle(.secondary)

            Button(action: toggleRecording) {
                Text(buttonTitle)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(buttonColor)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }
            .disabled(isTranscribing)

            if case let .finished(text) = service.state {
                ScrollView {
                    Text(text)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(height: 240)
            }

            Spacer()
        }
        .padding()
        .onReceive(service.$state) { state in
            status = statusMessage(for: state)
        }
    }

    private var buttonTitle: String {
        switch service.state {
        case .recording: return "Stop Recording"
        case .transcribing: return "Transcribing…"
        default: return "Start Recording"
        }
    }

    private var buttonColor: Color {
        switch service.state {
        case .recording: return .red
        case .transcribing: return .gray
        default: return .blue
        }
    }

    private var isTranscribing: Bool {
        if case .transcribing = service.state { return true }
        return false
    }

    private func toggleRecording() {
        switch service.state {
        case .recording:
            service.stopRecordingAndTranscribe()
        default:
            service.startRecording()
        }
    }

    private func statusMessage(for state: TranscriberService.State) -> String {
        switch state {
        case .idle: return "Ready"
        case .recording: return "Recording…"
        case .transcribing: return "Running Whisper…"
        case .finished: return "Done"
        case .failed(let error): return "Error: \(error.localizedDescription)"
        }
    }
}

#Preview {
    ContentView()
}
