type Props = { src: string; running: boolean; chassisVisible: boolean };

export function VideoPanel({ src, running, chassisVisible }: Props) {
  return (
    <section className="video card" aria-label="Live video">
      {running ? <img src={src} alt="Annotated camera view" /> : <div className="video-empty">Pick a source and start a run</div>}
      {running && !chassisVisible && <div className="banner">Chassis not visible</div>}
    </section>
  );
}
