export default function Loading() {
  return (
    <div className="min-h-[40vh] flex items-center justify-center p-6">
      <div className="animate-pulse space-y-4 w-full max-w-6xl">
        <div className="h-8 w-40 bg-gray-200 rounded" />
        <div className="h-40 bg-gray-200 rounded-2xl" />
        <div className="h-40 bg-gray-200 rounded-2xl" />
      </div>
    </div>
  );
}
