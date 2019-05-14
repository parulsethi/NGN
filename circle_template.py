import argparse

from PIL import Image, ImageDraw

DEFAULT_SIZE = 600
DEFAULT_CIRCLES = 10


class CircleTemplate:
    """
    Draws a circle template
    """

    def __init__(self, size, circle_count):
        self.size = size
        self.circle_count = circle_count
        self.image = Image.new(mode='L', size=(size, size), color=255)
        self.midpoint = int(size / 2)
        self._draw()
        self.save()

    def save(self):
        """Write my circle template image to file"""
        filename = "circle-{}-{}.png".format(self.size, self.circle_count)
        print("Saving {}".format(filename))
        self.image.save(filename)

    def show(self):
        """Display my circle template image on screen"""
        self.image.show()

    def _draw(self):
        """Create circles and slices in-memory"""
        draw = ImageDraw.Draw(self.image)
        # largest_circle = 
        self._draw_circles(draw)
        # self._draw_slices(draw, largest_circle)
        del draw

    def _draw_circles(self, draw):
        if self.circle_count <= 0:
            return 0

        radius_step = int(self.midpoint / self.circle_count)

        for radius in range(0, self.midpoint, radius_step):
            bounding_box = [
                (self.midpoint - radius, self.midpoint - radius),
                (self.midpoint + radius, self.midpoint + radius)]
            draw.arc(bounding_box, 0, 360)


def main():
    """Create a circle template from command line options"""
    # Get details from command line or use defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", help="length of image side in pixels",
                        type=int, default=DEFAULT_SIZE)
    parser.add_argument("--circles", help="number of circles",
                        type=int, default=DEFAULT_CIRCLES)
    args = parser.parse_args()
    size = args.size
    circle_count = args.circles
    circle_template = CircleTemplate(size, circle_count)
    circle_template.show()

if __name__ == '__main__':
    main()