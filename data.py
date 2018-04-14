import random

class BaseBatchGenerator:
    def generate_forever(self, items, batch_size):
        item = items[:]  # copy so we can shuffle in place
        while True:
            random.shuffle(items)
            yield from self.generate_epoch(items, batch_size)
            
    def generate_epoch(self, items, batch_size):
        steps = []
        for item in items:
            training_examples = self.process_item(item)
            steps.extend(training_examples)
            n_batches = len(steps) // batch_size
            n_steps = n_batches * batch_size
            batches = self.generate_batches(steps[:n_steps], batch_size, n_batches)
            steps = steps[n_steps:]
            yield from batches

    def generate_batches(self, steps, batch_size, n_batches):
        raise NotImplementedError

    def process_item(self, item):
        raise NotImplementedError
