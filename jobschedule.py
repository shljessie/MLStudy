import time
import threading
import queue

# Job data structure
class Job:
    def __init__(self, job_id, priority, required_resources):
        self.job_id = job_id
        self.priority = priority
        self.required_resources = required_resources  # E.g., {"cpu": 1, "gpu": 0, "memory": 2}
        self.status = "waiting"

# Resource Tracker
class ResourceTracker:
    def __init__(self, total_resources):
        self.available_resources = total_resources
        self.lock = threading.Lock()

    def allocate_resources(self, required_resources):
        with self.lock:
            for key in required_resources:
                if self.available_resources[key] < required_resources[key]:
                    return False
            for key in required_resources:
                self.available_resources[key] -= required_resources[key]
            return True

    def release_resources(self, required_resources):
        with self.lock:
            for key in required_resources:
                self.available_resources[key] += required_resources[key]

# Job Scheduler
class JobScheduler:
    def __init__(self, total_resources):
        self.job_queue = queue.PriorityQueue()  # Priority queue
        self.resource_tracker = ResourceTracker(total_resources)

    def submit_job(self, job):
        print(f"Submitting job {job.job_id} with priority {job.priority}")
        self.job_queue.put((job.priority, job))

    def dispatch_jobs(self):
        while True:
            if not self.job_queue.empty():
                priority, job = self.job_queue.get()
                print(f"Dispatching job {job.job_id}")
                allocated = self.resource_tracker.allocate_resources(job.required_resources)

                if allocated:
                    job.status = "running"
                    threading.Thread(target=self.execute_job, args=(job,)).start()
                else:
                    print(f"Insufficient resources for job {job.job_id}, re-queuing")
                    self.job_queue.put((priority, job))
            time.sleep(1)  # Check queue every second

    def execute_job(self, job):
        print(f"Executing job {job.job_id}")
        time.sleep(5)  # Simulate job running
        job.status = "completed"
        print(f"Job {job.job_id} completed")
        self.resource_tracker.release_resources(job.required_resources)

# Example usage
total_resources = {"cpu": 4, "gpu": 1, "memory": 8}  # Define cluster resources
scheduler = JobScheduler(total_resources)

# Submit jobs with various resource requirements
scheduler.submit_job(Job(job_id="job1", priority=1, required_resources={"cpu": 2, "gpu": 1, "memory": 4}))
scheduler.submit_job(Job(job_id="job2", priority=2, required_resources={"cpu": 1, "gpu": 0, "memory": 2}))
scheduler.submit_job(Job(job_id="job3", priority=3, required_resources={"cpu": 3, "gpu": 0, "memory": 2}))

# Start dispatching jobs
threading.Thread(target=scheduler.dispatch_jobs).start()
