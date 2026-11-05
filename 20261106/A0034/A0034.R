# Load R packages
suppressMessages(suppressWarnings(library(glue)))
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(ggplot2)))

# Set Env
no <- basename(dirname(rstudioapi::getActiveDocumentContext()$path))
wkdir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wkdir)

# Load data
DF <- readxl::read_excel('data/data.xlsx')
DF$value <- DF$value - 0.5

empty_bar <- 2 
DF$group <- as.factor(DF$group) 
to_add <- data.frame(matrix(NA, empty_bar*nlevels(DF$group), ncol(DF)))
colnames(to_add) <- colnames(DF)
to_add$group <- rep(levels(DF$group), each = empty_bar)
DF <- rbind(DF, to_add)
DF <- DF %>% arrange(group)
DF$id <- seq(1, nrow(DF))
DF <- DF %>% arrange(group, value)
DF$id <- seq(1, nrow(DF))

label_data <- DF
number_of_bar <- nrow(label_data)
angle <- 90 - 360 * (label_data$id - 0.5) / number_of_bar 
label_data$hjust <- ifelse( angle < -90, 1, 0)
label_data$angle <- ifelse(angle < -90, angle + 180, angle)

base_data <- DF %>% 
  group_by(group) %>% 
  summarize(start = min(id), end = max(id) - empty_bar) %>%
  rowwise() %>% 
  mutate(title = mean(c(start, end)))

grid_data <- base_data
grid_data$end <- grid_data$end[ c(nrow(grid_data), 1:nrow(grid_data) - 1)] + 0.7
grid_data$start <- grid_data$start - 0.7
grid_data <- grid_data[-1, ]

# Plotting constants
base.size   <- 7
text.size   <- 5 / ggplot2::.pt
base.family <- 'sans'

# Draw plot
gg <- ggplot(DF, aes(x = as.factor(id), y = -value, fill = group))
gg <- gg + geom_bar(stat = 'identity', alpha = 0.3) 
gg <- gg + scale_fill_manual(values = c("#3B4992FF", "#EE0000FF", "#008B45FF", "#631879FF"))
gg <- gg + ylim(-1.03, 0.17)
gg <- gg + geom_segment(data = grid_data, aes(x = end, y = -0.5, xend = start, yend = -0.5), colour = 'black', alpha = 1, linewidth = 0.8, inherit.aes = FALSE)
gg <- gg + geom_segment(data = grid_data, aes(x = end, y = -0.4, xend = start, yend = -0.4), colour = 'black', alpha = 1, linewidth = 0.8, inherit.aes = FALSE)
gg <- gg + geom_segment(data = grid_data, aes(x = end, y = -0.3, xend = start, yend = -0.3), colour = 'black', alpha = 1, linewidth = 0.8 , inherit.aes = FALSE)
gg <- gg + geom_segment(data = grid_data, aes(x = end, y = -0.2, xend = start, yend = -0.2), colour = 'black', alpha = 1, linewidth = 0.8 , inherit.aes = FALSE)
gg <- gg + geom_segment(data = grid_data, aes(x = end, y = -0, xend = start, yend = -0.), colour = 'black', alpha = 1, linewidth = 0.8 , inherit.aes = FALSE)
gg <- gg + annotate('text', x = rep(23.8, 5), y = c(-0.5, -0.4, -0.3, -0.2, 0.0) - 0.045, label = c('1.0', '0.9', '0.8', '0.7', '0.5'), color = '#000000', size = text.size , angle = 0, fontface = 'bold', hjust = 0, family = base.family)
gg <- gg + geom_segment(data = grid_data, aes(x = 1, y = -0.3, xend = 48.5, yend = -0.3), linetype = 'dashed',colour = 'red', alpha = 0.1, linewidth = 0.5 , inherit.aes = FALSE)
gg <- gg + geom_segment(data = grid_data, aes(x = 1, y = -0.5, xend = 48.5, yend = -0.5), linetype = 'dashed',colour = 'red', alpha = 0.1, linewidth = 0.5 , inherit.aes = FALSE)
gg <- gg + theme_minimal(base_size = base.size, header_family = base.family)
gg <- gg + theme(
  legend.position = c(.5, .5),
  legend.title = element_text(size = base.size),
  legend.text = element_text(size = base.size - 1),
  legend.key.height = unit(0.15, "cm"),
  legend.key.width  = unit(0.15, "cm"),
  legend.spacing.x  = unit(0.15, "cm"),
  legend.key.spacing.y = unit(0.12, 'cm'),
  axis.text = element_blank(),
  axis.title = element_blank(),
  panel.grid = element_blank()
)
gg <- gg + labs(fill = 'Indicator')
gg <- gg + coord_polar()
gg <- gg + geom_segment(data = base_data, aes(x = start - 0.7, y = 0.007, xend = end + 0.7, yend = 0.007), colour = 'black', alpha = 0.5, linewidth = 2 , inherit.aes = FALSE)
gg <- gg + geom_segment(data = base_data, aes(x = start - 0.7, y = 0.15, xend = end + 0.7, yend = 0.15), colour = c("#3B4992FF", "#EE0000FF", "#008B45FF", "#631879FF"), alpha = 0.35, linewidth = 61 , inherit.aes = FALSE)
gg <- gg + geom_text(data = label_data, aes(x = id, y = 0.05, label = individual, hjust = hjust), color = 'black', fontface = 'bold', alpha = ifelse(label_data$value > 0.045, 0.9, 0.3), size = text.size, angle = label_data$angle, inherit.aes = FALSE, family = base.family)
gg <- gg + geom_text(data = label_data, aes(x = id, y = value * -1 + 0.015, label = round(value + 0.5, 2), hjust = hjust), alpha = 1, size = text.size, angle = label_data$angle, inherit.aes = FALSE, family = base.family)
  
  
# Save to file
width = 90; height = 90
ggsave(gg, filename = glue('{wkdir}/{no}.png'), width = width, height = height, units = 'mm', dpi = 300, device = 'png', bg = '#FFFFFF')
ggsave(gg, filename = glue('{wkdir}/{no}.pdf'), width = width, height = height, units = 'mm', dpi = 300, device = cairo_pdf, bg = '#FFFFFF')

# sessionInfo
sink(glue('{wkdir}/{no}.txt'))
sessionInfo()
sink()



