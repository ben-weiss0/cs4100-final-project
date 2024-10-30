import pygame


def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)


def blit_rotate_center(win, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(
        center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)

def rotate_center(image, top_left, angle):
    # Rotate the image
    rotated_image = pygame.transform.rotate(image, angle)
    # Get the new rectangle that centers the rotated image based on the original top_left position
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    return rotated_image, new_rect

def blit_text_center(win, font, text):
    render = font.render(text, 1, (200, 200, 200))
    win.blit(render, (win.get_width()/2 - render.get_width() /
                      2, win.get_height()/2 - render.get_height()/2))

def export_window():
    pygame.image.save(WIN, 'state.png')